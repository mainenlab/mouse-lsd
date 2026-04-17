import os
import shutil
import tarfile
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

import bombcell

from one.api import ONE
from one.alf.exceptions import ALFObjectNotFound

from psyfun.io import load_units

class SpikeSortingQC():

    def __init__(self, eid, probe, one):
        self.eid = eid
        self.probe = probe
        self.one = one

        # Try to see if KS output is already downloaded and extracted
        sorter = self._get_sorter()
        self.spike_sorting_dir = one.eid2path(self.eid) / f'spike_sorters/{probe}/{sorter}'
        # Pick a random KS output file to check for
        sorting_extracted = (self.spike_sorting_dir / 'amplitudes.npy').exists()

        if not sorting_extracted:
            self.spike_sorting_filepath = self.download_spike_sorting()
            # Overwrite dir with one that actually had the KS output archive
            self.spike_sorting_dir = self.spike_sorting_filepath.parent
            self.extract_spike_sorting()


    def _get_sorter(self):
        collections = self.one.list_collections(self.eid)
        has_pykilosort = any(
            ['pykilosort' in col for col in collections if self.probe in col]
        )
        has_iblsorter = any(
            ['iblsorter' in col for col in collections if self.probe in col]
            )
        if has_iblsorter:  # priority given to iblsorter
            sorter = 'iblsorter'
        elif has_pykilosort:
            sorter = 'pykilosort'
        else:
            sorter = None


    def download_spike_sorting(self):
        print("Loading spike sorting data...")
        try:
            spike_sorting_filepath = self.one.load_dataset(
                eid,
                '*_kilosort_raw.output.tar',
                collection=f'spike_sorters/pykilosort/{probe}'
            )
        except ALFObjectNotFound:
            spike_sorting_filepath = self.one.load_dataset(
                eid,
                '*_kilosort_raw.output.tar',
                collection=f'spike_sorters/iblsorter/{probe}'
            )
        return spike_sorting_filepath


    def extract_spike_sorting(self):
        with tarfile.open(self.spike_sorting_filepath, "r:*") as tar:
            self._safe_tar_extract(tar, self.spike_sorting_dir)


    def remove_spike_sorting(self):
        print("Deleting spike sorting archive...")
        spike_sorting_filepath = self.spike_sorting_dir / '_kilosort_raw.output.tar'
        try:
            os.remove(spike_sorting_filepath)
            print("    spike sorting archive deleted.")
        except FileNotFoundError:
            print(f"    spike sorting archive not found: {spike_sorting_filepath}")


    def _safe_tar_extract(self, tar, path):
        print("Extracting spike sorting data...")
        for member in tar.getmembers():
            member_path = os.path.join(path, member.name)
            if not os.path.realpath(member_path).startswith(os.path.realpath(path)):
                raise Exception("Unsafe path detected")
        tar.extractall(path)


    def load_waveforms(self):
        try:
            waveforms = self.one.load_dataset(
                self.eid,
                '*clusters.waveforms.py',
                collection=f'alf/{self.probe}/pykilosort'
            )
            channels = self.one.load_dataset(
                self.eid,
                'clusters.waveformsChannels.npy',
                collection=f'alf/{self.probe}/pykilosort'
                )
        except ALFObjectNotFound:
            waveforms = self.one.load_dataset(
                self.eid,
                '*clusters.waveforms.py',
                collection=f'alf/{self.probe}/iblsorter'
            )
            channels = self.one.load_dataset(
                self.eid,
                'clusters.waveformsChannels.npy',
                collection=f'alf/{self.probe}/iblsorter'
                )
        return waveforms, channels


    def run_bombcell(self) -> pd.DataFrame:
        print("Preparing bombcell...")
        self.bombcell_dir = self.spike_sorting_dir / 'bombcell'
        if os.path.exists(self.bombcell_dir):
            print("Deleting previous bombcell output...")
            shutil.rmtree(self.bombcell_dir)
        os.mkdir(self.spike_sorting_dir / 'bombcell')

        param = bombcell.get_default_parameters(
            kilosort_path=self.spike_sorting_dir,
            kilosort_version=''  # '4' if KS4, else anything
        )

        print("Running bombcell...")
        quality_metrics, param, unit_type, unit_type_string = bombcell.run_bombcell(
            ks_dir=self.spike_sorting_dir,
            save_path=self.spike_sorting_dir / 'bombcell',
            param=param,
            save_figures=True
        )

        plt.close('all')
        print(f"    results saved to {self.bombcell_dir}")

        bombcell_results = pd.DataFrame.from_dict(quality_metrics)
        bombcell_results['label'] = unit_type_string

        self.bombcell_param = param
        self.bombcell_results = bombcell_results

        return bombcell_results


    def attach_uuid(self, probe_units: pd.DataFrame) -> pd.DataFrame:
        """Merge `uuid` from the units table onto `self.bombcell_results` by
        `cluster_id`, verifying that spike counts match for every matched cluster."""
        merged = self.bombcell_results.merge(
            probe_units[['cluster_id', 'uuid', 'spike_count']],
            left_on='phy_clusterID',
            right_on='cluster_id',
            how='inner',
            validate='one_to_one',
        )
        if not np.array_equal(merged['nSpikes'].to_numpy(), merged['spike_count'].to_numpy()):
            raise ValueError(
                f"Spike counts disagree between bombcell and units for "
                f"eid={self.eid}, probe={self.probe}"
            )
        self.bombcell_results = merged.drop(columns=['cluster_id', 'spike_count'])
        return self.bombcell_results


if __name__ == '__main__':
    one = ONE()

    df_units = load_units()

    probes = df_units.groupby(['eid', 'probe'])
    for (eid, probe), probe_units in tqdm(probes, total=len(probes)):
        ssqc = SpikeSortingQC(eid, probe, one)
        ssqc.run_bombcell()
        bombcell_results = ssqc.attach_uuid(probe_units)
        ssqc.remove_spike_sorting()

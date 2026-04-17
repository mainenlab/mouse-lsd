import pandas as pd
from tqdm import tqdm

from one.api import ONE

from psyfun.config import paths
from psyfun.io import load_units
from psyfun.spike_sorting import SpikeSortingQC


if __name__ == '__main__':
    one = ONE()

    df_units = load_units()

    all_results = []
    probes = df_units.groupby(['eid', 'probe'])
    for (eid, probe), probe_units in tqdm(probes, total=len(probes)):
        ssqc = SpikeSortingQC(eid, probe, one)
        ssqc.run_bombcell()
        bombcell_results = ssqc.attach_uuid(probe_units)
        all_results.append(bombcell_results)
        ssqc.remove_spike_sorting()

    df_bombcell = pd.concat(all_results, ignore_index=True)
    df_bombcell.to_parquet(paths['bombcell'], index=False)
    print(f"Saved {len(df_bombcell)} units to {paths['bombcell']}")

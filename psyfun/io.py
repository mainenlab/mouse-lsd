import os
import re
from functools import lru_cache
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
tqdm.pandas()
import warnings
import h5py

from one.alf.exceptions import ALFObjectNotFound
from brainbox.io.one import SpikeSortingLoader
from iblatlas.atlas import AllenAtlas
atlas = AllenAtlas()

from psyfun.atlas import region_parcellation
from psyfun.config import paths, df_controls, TASKTIMINGS

PASSIVE_PROTOCOL_TOKEN = 'passive'
SPONTANEOUS_PROTOCOL_TOKEN = 'spontaneous'
PROTOCOL_REPLAY_DURATION_S = 300.0  # protocol design: 5 min gabor replay
RAW_TASK_COLLECTION_RE = re.compile(r'^raw_task_data_(\d+)$')


def fetch_sessions(one, save=True, qc=False):
    """
    Query Alyx for sessions tagged in the psychedelics project and add session
    info to a dataframe. Sessions are restricted to those with the
    passiveChoiceWorld task protocol, quality control metadata is unpacked, and
    a list of key datasets is checked. Sessions are sorted and labelled
    (session_n) by their order.

    Parameters
    ----------
    one : one.api.OneAlyx
        Alyx database connection instance
    save : bool
        If true, the dataframe will be saved in ./metadata/sessions.csv

    Returns
    -------
    df_sessions : pandas.core.frame.DataFrame
        Dataframe containing detailed info for each session returned by the
        query
    """
    # Query for all sessions in the project with the specified task
    sessions = one.alyx.rest('sessions', 'list', project=ibl_project['name'], task_protocol=ibl_project['protocol'])
    df_sessions = pd.DataFrame(sessions).rename(columns={'id': 'eid'})
    df_sessions.drop(columns='projects')
    if qc:
        # Unpack the extended qc from the session dict into dataframe columns
        print("Unpacking quality control data...")
        df_sessions = df_sessions.progress_apply(_unpack_session_dict, one=one, axis='columns').copy()
    # Check if important datasets are present for the session
    df_sessions['n_probes'] = df_sessions.apply(lambda x: len(one.eid2pid(x['eid'])[0]), axis='columns')
    df_sessions['n_tasks'] = df_sessions['task_protocol'].apply(lambda x: sum(['passive' in task.lower() for task in x.split('_')]))
    df_sessions['tasks'] = df_sessions.apply(lambda x: x['task_protocol'].split('/'), axis='columns')
    print("Checking datasets...")
    df_sessions = df_sessions.progress_apply(_check_datasets, one=one, axis='columns').copy()
    # Add label for control sessions
    df_sessions['control_recording'] = df_sessions.apply(_label_controls, axis='columns')
    df_sessions['new_recording'] = df_sessions['start_time'].apply(lambda x: datetime.fromisoformat(x) > datetime(2025, 1, 1))
    # Add label for the electrode insertion trajectories
    df_sessions = get_trajectory_labels(df_sessions)
    # Fetch task protocol timings and add to dataframe
    print("Fetching protocol timings...")
    df_sessions = df_sessions.progress_apply(_fetch_protocol_timings, one=one, axis='columns').copy()
    # Add LSD administration time
    df_meta = load_metadata()
    df_sessions = df_sessions.apply(_insert_LSD_admin_time, df_metadata=df_meta, axis='columns').copy()
    # Label and sort by session number for each subject
    df_sessions['session_n'] = df_sessions.groupby('subject')['start_time'].rank(method='dense').astype(int)
    df_sessions = df_sessions.sort_values(by=['subject', 'start_time']).reset_index(drop=True)
    # Save as csv
    if save:
        df_sessions.to_parquet(paths['sessions'], index=False)
    return df_sessions


@lru_cache(maxsize=1)
def _get_default_connection():
    """
    Create and cache the default database connection. Cached connection allows
    repeated function calls without re-creating connection instance.
    """
    return ONE()


def _unpack_session_dict(series, one=None):
    """
    Unpack the extended QC from the session dict for a given eid.
    """
    if one is None:
        one = _get_default_connection()
    # Fetch full session dict
    session_dict = one.alyx.rest('sessions', 'read', id=series['eid'])
    series['session_qc'] = session_dict['qc']  # aggregate session QC value
    # Skip if there is no extended QC present
    if session_dict['extended_qc'] is None:
        return series
    # Add QC vals to series
    for key, val in session_dict['extended_qc'].items():
        key = key.lstrip('_')
        # Add _qc flag to any keys that don't have it
        if not key.endswith('_qc'): key += '_qc'
        if type(val) == str:
           series[key] = val
        elif type(val) == list:  # lists have QC outcome as first entry
            series[key] = val[0]
            # Add video framerate
            if 'framerate' in key:
                series[key.rstrip('_qc')] = val[1]
            # Add number of dropped frames
            if 'dropped_frames' in key:
                series[key.rstrip('_qc')] = val[1]
    return series


def _check_datasets(series, one=None):
    """
    Create a boolean entry for each important dataset for the given eid.
    """
    if one is None:
        one = _get_default_connection()
    # Fetch list of datasets listed under the given eid
    datasets = one.list_datasets(series['eid'])
    # Check each task in the recording
    for n, protocol in enumerate(series['tasks']):
        # TODO: handle spontaneous protocol in 2025 recordings!
        if 'spontaneous' in protocol:
            continue
        for dataset in qc_datasets['task']:
            series[dataset] = dataset in datasets
    for n in range(series['n_probes']):
        for dataset in qc_datasets['ephys']:
            series[dataset] = dataset in datasets
    # Check if each important dataset is present
    for dataset in qc_datasets['video']:
        series[dataset] = dataset in datasets
    return series


def _label_controls (session, controls=df_controls):
    eid = session['eid']
    control_session = controls.query('eid == @eid')
    if len(control_session) == 1:
        return True
    elif len(control_session) == 0:
        return False
    elif len(control_session) > 1:
        raise ValueError("More than one entry in df_controls!")


def _list_raw_task_collections(eid: str, one) -> list[str]:
    """All `raw_task_data_NN` collections present for `eid`, sorted by NN."""
    matches = {
        m.group(0): int(m.group(1))
        for d in one.list_datasets(eid)
        for m in [RAW_TASK_COLLECTION_RE.match(d.split('/')[0])]
        if m
    }
    return [c for c, _ in sorted(matches.items(), key=lambda kv: kv[1])]


def _rig_session_datetime(settings: dict) -> datetime:
    """Local-clock SESSION_DATETIME (or SESSION_START_TIME) from iblrig settings."""
    raw = settings.get('SESSION_DATETIME') or settings.get('SESSION_START_TIME')
    if raw is None:
        raise KeyError("Neither 'SESSION_DATETIME' nor 'SESSION_START_TIME' in taskSettings")
    return datetime.fromisoformat(raw)


def _fpga_timings_from_alf(intervals: pd.DataFrame, gabor: pd.DataFrame | None) -> dict:
    """
    Build FPGA-aligned epoch boundaries from canonical alf datasets.

    Uses `intervalsTable` for spontaneous/RFM (both columns valid), and
    `passiveGabor.start.min()/.stop.max()` for replay if available;
    otherwise falls back to `intervalsTable.taskReplay[0]` plus the
    fixed 5-min protocol replay duration. The intervalsTable's
    `taskReplay` stop is unreliable (extends to end of recording).
    """
    if 'Unnamed: 0' in intervals.columns:
        intervals = intervals.set_index('Unnamed: 0')
    elif intervals.index.name != '':
        intervals = intervals.set_index(intervals.columns[0])
    spont_start, spont_stop = intervals['spontaneousActivity'].iloc[:2].to_list()
    rfm_start, rfm_stop = intervals['RFM'].iloc[:2].to_list()
    if gabor is not None and len(gabor) > 0:
        replay_start = float(gabor['start'].min())
        replay_stop = float(gabor['stop'].max())
    else:
        replay_start = float(intervals['taskReplay'].iloc[0])
        replay_stop = replay_start + PROTOCOL_REPLAY_DURATION_S
    return {
        'spontaneous_start': float(spont_start),
        'spontaneous_stop': float(spont_stop),
        'rfm_start': float(rfm_start),
        'rfm_stop': float(rfm_stop),
        'replay_start': replay_start,
        'replay_stop': replay_stop,
    }


def _shift_timings(anchor: dict, anchor_rig_t0: datetime, target_rig_t0: datetime) -> dict:
    """Translate a passive run's FPGA timings by the rig wall-clock delta."""
    delta_s = (target_rig_t0 - anchor_rig_t0).total_seconds()
    return {k: v + delta_s for k, v in anchor.items()
            if k in {'spontaneous_start', 'spontaneous_stop',
                     'rfm_start', 'rfm_stop',
                     'replay_start', 'replay_stop'}}


def _load_passive_run(eid: str, raw_col: str, one) -> dict | None:
    """
    Load FPGA-aligned timings + rig anchor for one passive `raw_task_data_NN`.

    Returns None if the protocol in this collection is not passive. If alf
    data is missing, returns dict with `alf_missing=True` so the caller
    can fill via rig-clock fallback.
    """
    settings = one.load_dataset(eid, '_iblrig_taskSettings.raw.json', raw_col)
    protocol = settings.get('PYBPOD_PROTOCOL', '')
    if SPONTANEOUS_PROTOCOL_TOKEN in protocol or PASSIVE_PROTOCOL_TOKEN not in protocol:
        return None
    rig_t0 = _rig_session_datetime(settings)
    alf_col = raw_col.replace('raw_task_data_', 'alf/task_')
    try:
        intervals = one.load_dataset(eid, '_ibl_passivePeriods.intervalsTable.csv', alf_col)
    except ALFObjectNotFound:
        return {'rig_t0': rig_t0, 'alf_missing': True}
    try:
        gabor = one.load_dataset(eid, '_ibl_passiveGabor.table.csv', alf_col)
    except ALFObjectNotFound:
        gabor = None
    return {**_fpga_timings_from_alf(intervals, gabor), 'rig_t0': rig_t0}


def _fetch_protocol_timings(series, one=None):
    """
    Populate `task_NN_<epoch>_<start|stop>` columns in spike-time seconds.

    Iterates `raw_task_data_NN` collections, skips spontaneous-only filler
    protocols, and reads FPGA-aligned timings from
    `alf/task_NN/_ibl_passivePeriods.intervalsTable.csv` and
    `_ibl_passiveGabor.table.csv`. For passive runs without alf data,
    derives timings from another extracted passive run on the same
    session via the iblrig wall-clock delta between
    `SESSION_DATETIME`s. See `specs/fix_dst_task_timings.md`.
    """
    if one is None:
        one = _get_default_connection()
    eid = series['eid']
    for col in TASKTIMINGS:
        if col != 'LSD_admin':
            series[col] = np.nan
    timings: list[dict] = []
    for raw_col in _list_raw_task_collections(eid, one):
        run = _load_passive_run(eid, raw_col, one)
        if run is not None:
            timings.append(run)
    anchor = next((t for t in timings if not t.get('alf_missing')), None)
    if anchor is None:
        return series
    for passive_idx, run in enumerate(timings):
        if run.get('alf_missing'):
            run = {**_shift_timings(anchor, anchor['rig_t0'], run['rig_t0']), 'rig_t0': run['rig_t0']}
        for epoch in ('spontaneous', 'rfm', 'replay'):
            for endpoint in ('start', 'stop'):
                series[f'task{passive_idx:02d}_{epoch}_{endpoint}'] = run[f'{epoch}_{endpoint}']
    return series


def _insert_LSD_admin_time(series, df_metadata=None):
    assert df_metadata is not None
    # Find entry in metadata file by subject and date
    session_meta = df_metadata[
        (df_metadata['animal_ID'] == series['subject']) &
        (df_metadata['date'] == datetime.fromisoformat(series['start_time']).date())
    ]
    # Ensure only one entry is present
    if len(session_meta) < 1:
        warnings.warn(f"No entries in 'metadata.csv' for {series['eid']}")
        return series
    elif len(session_meta) > 1:
        warnings.warn(f"More than one entry in 'metadata.csv' for {series['eid']}")
        return series
    series['LSD_admin'] = session_meta['administration_time'].values[0]
    return series


def get_trajectory_labels(df_sessions, drop=True, hemisphere=True):
    df_trajectories = pd.read_csv(paths['trajectories'])
    df_sessions = df_sessions.apply(
        _insert_trajectory_labels,
        df_trajectories=df_trajectories,
        axis='columns'
        ).copy()
    if drop:
        df_sessions = df_sessions.dropna(subset=['trajectory_01', 'trajectory_02'])
    if hemisphere:
        combine_labels = lambda x: '_'.join([
            str(x['trajectory_01']),
            str(x['trajectory_02'])
            ])
    else:
        combine_labels = lambda x: '_'.join([
            str(x['trajectory_01']).rstrip('L').rstrip('R'),
            str(x['trajectory_02']).rstrip('L').rstrip('R')
            ])
    df_sessions['trajectory_label'] = df_sessions.apply(
        combine_labels,
        axis='columns'
    )
    return df_sessions


def _insert_trajectory_labels(series, df_trajectories):
    # assert df_trajectories is not None
    eid = series['eid']
    trajectories = df_trajectories.query('eid == @eid')
    if len(trajectories) == 1:
        trajectories = trajectories.iloc[0]
        for col in trajectories.index:
            if col in ['date', 'subject', 'eid']:
                continue
            series[col] = trajectories[col]
    elif len(trajectories) > 1:
        raise ValueError(f"Multiple trajectory entries found for eid {eid}")
    return series


def fetch_insertions(one, save=True):
    """
    Query Alyx for probe insertions tagged in the psychedelics project and
    collect insertion info in a dataframe.

    Parameters
    ----------
    one : one.api.OneAlyx
        Alyx database connection instance
    save : bool
        If true, the dataframe will be saved in ./metadata/sessions.csv

    Returns
    -------
    df_insertions : pandas.core.frame.DataFrame
        Dataframe containing detailed info for each probe insertion returned by
        the query
    """
    # Query for all probe insertions in the project
    insertions = one.alyx.rest('insertions', 'list', project=ibl_project['name'])
    df_insertions = pd.DataFrame(insertions).rename(columns={'id': 'pid', 'session': 'eid', 'name':'probe'})
    # Pull out some basic fields from the session info dict
    df_insertions = df_insertions.progress_apply(_unpack_session_info, axis='columns').copy()
    # Unpack detailed QC info from the json
    df_insertions = df_insertions.progress_apply(_unpack_json, axis='columns').copy()
    # Add any histology QC info present
    df_insertions = df_insertions.progress_apply(_check_histology, one=one, axis='columns').copy()
    # Label and sort by session number for each subject
    df_insertions['session_n'] = df_insertions.groupby('subject')['start_time'].rank(method='dense').astype(int)
    df_insertions = df_insertions.sort_values(by=['subject', 'start_time']).reset_index(drop=True)
    # Save as csv
    if save:
        df_insertions.to_parquet(paths['insertions'], index=False)
    return df_insertions


def _unpack_session_info(series):
    series['subject'] = series['session_info']['subject']
    series['start_time'] = series['session_info']['start_time']
    return series


def _unpack_json(series):
    if not series['json']:
        print(f"WARNING: ephys qc json empty for pid {series['pid']}")
        return series
    series['ephys_qc'] = series['json']['qc']
    jsonkeys = ['n_units', 'n_units_qc_pass', 'firing_rate_median', 'firing_rate_max']
    for key in jsonkeys:
        try:
            series[key] = series['json'][key]
        except KeyError:
            series[key] = np.nan
    if 'tracing_exists' not in series['json']['extended_qc']:
        series['tracing_qc'] = 'NOT SET'
        series['alignment_qc'] = 'NOT SET'
    elif series['json']['extended_qc']['tracing_exists']:
        if 'tracing' in series['json']['extended_qc']:
            series['tracing_qc'] = series['json']['extended_qc']['tracing']
        else:
            series['tracing_qc'] = 'NOT SET'
        try:
            alignment_resolved_by = series['json']['extended_qc']['alignment_resolved_by']
            series['alignment_qc'] = series['json']['extended_qc'][alignment_resolved_by]
        except KeyError:
            series['alignment_qc'] = 'NOT SET'
    else:
        series['tracing_qc'] = series['json']['extended_qc'].get('tracing', 'NOT SET')
        series['alignment_qc'] = 'NOT SET'
    return series


def _check_histology(series, one=None):
    assert one is not None
    infos = np.array(one.alyx.rest('sessions', 'list', subject=series['subject'], histology=True))
    histology_in_protocols = ['histology' in info['task_protocol'].lower() for info in infos]
    if any(histology_in_protocols):
        ## NOT SET for all...
        histology_eid = infos[histology_in_protocols][0]['id']
        hist_dict = one.alyx.rest('sessions', 'read', id=histology_eid)
        series['histology_qc'] = hist_dict['qc']
        # series['histology_qc'] = 'NOT SET'
    else:
        series['histology_qc'] = np.nan
    return series


def load_metadata():
    """
    Loads recording metadata .csv as a pandas DataFrame. Converts date column
    to a datetime object and administration time to seconds.
    """
    df_meta = pd.read_csv(paths['metadata'])
    df_meta['date'] = df_meta['date'].apply(lambda x: datetime.strptime(x, '%d.%m.%Y').date())
    hms2sec = lambda hms: np.sum(np.array([int(val) for val in hms.split(':')]) * np.array([3600, 60, 1]))
    df_meta['administration_time'] = df_meta['administration_time'].apply(hms2sec)
    return df_meta


def _datetime_clip_decimals_to_iso(datetime_str):
    main, decimals = datetime_str.split('.')
    decimals = decimals[:6]  # keep only 6 digits
    datetime_str = f"{main}.{decimals}"
    return datetime.fromisoformat(datetime_str)


class PsySpikeSortingLoader(SpikeSortingLoader):

    def merge_clusters(self, clusters, channels, compute_metrics=False, spikes=None):
        """
        A simplified method for merging metrics and channel info into the
        clusters dict. Does not require spikes to save memory, can be
        optionally given together with compute_metrics=True to re-compute
        quality control metrics on-the-fly.
        """
        nc = clusters['channels'].size
        metrics = None
        if 'metrics' in clusters:
            metrics = clusters.pop('metrics')
            if metrics.shape[0] != nc:
                metrics = None
        if metrics is None or compute_metrics is True:
            assert spikes is not None
            metrics = SpikeSortingLoader.compute_metrics(spikes, clusters)
        for k in metrics.keys():
            clusters[k] = metrics[k].to_numpy()
        for k in channels.keys():
            if k in ['localCoordinates', 'rawInd']: continue
            clusters[k] = channels[k][clusters['channels']]
        return clusters


def fetch_unit_info(one, df_insertions, uinfo_file='', spike_file='', atlas=atlas):
    probe_dfs = []
    for idx, probe in tqdm(df_insertions.iterrows(), total=len(df_insertions)):
        # Load in spike times and cluster info
        pid = probe['pid']
        loader = PsySpikeSortingLoader(pid=pid, one=one, atlas=atlas)
        try:
            clusters = loader.load_spike_sorting_object('clusters')
            channels = loader.load_channels()
        except KeyError:
            continue
        if clusters is None:
            print(f"WARNING: no clusters for {pid}")
            continue
        clusters['uuids'] = clusters['uuids'].to_numpy()  # take values out of dataframe
        clusters = loader.merge_clusters(clusters, channels)
        # Build dataframe from list for this probe
        df_probe = pd.DataFrame(clusters).rename(columns={'uuids':'uuid', 'depths':'depth', 'channels':'channel'})
        # Add additional metadata to cluster info df
        for field in ['subject', 'session_n', 'eid', 'probe', 'pid']:
            df_probe[field] = probe[field]
        df_probe['histology'] = loader.histology
        # Save spike times if a filename is given
        if spike_file:
            # Load spike time for each probe collection separately to conserve memory
            for collection in loader.collections:
                spikes = one.load_object(probe['eid'], collection=collection, obj='spikes', attribute=['times', 'clusters'])
                with h5py.File(spike_file, 'a') as h5file:
                    # One group per uuid; spike times under 'times', leaving room
                    # for sibling datasets (e.g. 'duplicates', 'waveforms').
                    for uuid, cid in zip(df_probe['uuid'], df_probe['cluster_id']):
                        spike_times = spikes['times'][spikes['clusters'] == cid]
                        if uuid in h5file: del h5file[uuid]
                        grp = h5file.create_group(uuid)
                        grp.create_dataset('times', data=spike_times)
                del spikes
        del clusters, loader
        # Append to list
        probe_dfs.append(df_probe)
    # Concatenate cluster info for all probes
    df_uinfo = pd.concat(probe_dfs)
    # Clean up some columns
    df_uinfo['histology'] = df_uinfo['histology'].fillna('')
    df_uinfo = df_uinfo.rename(columns={'acronym': 'region'})
    if uinfo_file:
        df_uinfo.to_parquet(uinfo_file, index=False)
    return df_uinfo


def load_spikes(uuids, remove_duplicates: bool = True) -> pd.DataFrame:
    """
    Load per-uuid spike times from the HDF5 file at `paths['spikes']`.

    Parameters
    ----------
    uuids : iterable of str
        Unit UUIDs to load. Each must exist as a group in the spikes h5 file.
    remove_duplicates : bool
        If True, read the per-uuid 'duplicates' bool mask and return
        `times[~mask]`. Uuids without a 'duplicates' dataset get raw times;
        a single aggregated warning is printed at the end.

    Returns
    -------
    df_spiketimes : pandas.DataFrame
        Dataframe indexed by uuid with a single 'spike_times' column holding
        a numpy array of spike times per unit.
    """
    units = []
    n_missing = 0
    with h5py.File(paths['spikes'], 'r') as h5file:
        for uuid in tqdm(uuids):
            grp = h5file[uuid]
            times = grp['times'][:]
            if remove_duplicates:
                if 'duplicates' in grp:
                    times = times[~grp['duplicates'][:]]
                else:
                    n_missing += 1
            units.append({'uuid': uuid, 'spike_times': times})
    if remove_duplicates and n_missing:
        print(
            f"WARNING: {n_missing}/{len(uuids)} uuids had no 'duplicates' "
            f"dataset; raw times returned for those"
        )
    return pd.DataFrame(units).set_index('uuid')


def save_duplicate_masks(masks: dict, spike_file) -> None:
    """Write per-uuid duplicate-spike masks into the spikes h5 file.

    Each `h5file[uuid]` must already be a group (created by the spike-saving
    block of the units-table builder). Writes the bool mask as
    `h5file[uuid]['duplicates']`, replacing any existing dataset.
    """
    with h5py.File(spike_file, 'a') as h5file:
        for uuid, mask in masks.items():
            grp = h5file[uuid]
            if 'duplicates' in grp: del grp['duplicates']
            grp.create_dataset('duplicates', data=mask.astype(bool))


def load_sessions(
    drop_if_nan: list = TASKTIMINGS,
    drop_extra_columns: bool = True,
) -> pd.DataFrame:
    """
    Load the sessions dataframe from `paths['sessions']`.

    Parameters
    ----------
    drop_if_nan : list of str
        Columns used to filter sessions: a session is dropped if any listed
        column is NaN. Defaults to task timing columns.
    drop_extra_columns : bool
        If True, drop QC and dataset columns and keep only identifiers,
        start time, control flag, and the full set of task timing columns
        (`TASKTIMINGS`, regardless of `drop_if_nan`).

    Returns
    -------
    df_sessions : pandas.DataFrame
        Sessions dataframe filtered by `drop_if_nan`.
    """
    df_sessions = pd.read_parquet(paths['sessions'])
    print(f"Total sessions: {len(df_sessions)}")
    # Remove sessions missing timing information for important experimental epochs
    df_sessions = df_sessions.dropna(subset=drop_if_nan)
    print(f"Valid sessions: {len(df_sessions)}")

    if drop_extra_columns:
        columns_to_keep = TASKTIMINGS + [
            'subject', 'eid', 'start_time', 'session_n', 'control_recording'
            ]
        df_sessions = df_sessions[columns_to_keep].copy()

    return df_sessions


def load_units(
    eids=None,
    unit_filter: str = None,
    add_coarse_regions: bool = True,
) -> pd.DataFrame:
    """
    Load the units dataframe from `paths['units']`.

    Parameters
    ----------
    eids : iterable of str or None
        If None, no session filter is applied. Otherwise, keep only units
        whose `eid` is in the provided collection (an empty collection
        returns zero rows).
    unit_filter : str or None
        Optional `pandas.DataFrame.query` expression used to filter units
        (e.g. quality-control thresholds). If None, no filter is applied.
    add_coarse_regions : bool
        If True, add a 'coarse_region' column via
        `psyfun.atlas.region_parcellation` applied to the 'region' column.

    Returns
    -------
    df_units : pandas.DataFrame
        Units dataframe after optional session and quality filters.
    """
    df_units = pd.read_parquet(paths['units'])

    if eids is not None:
        df_units = df_units.query('eid in @eids')
    print(f"N units: {len(df_units)}")

    # Remove low-quality units
    if unit_filter:
        df_units = df_units.query(unit_filter)
        print(f"    Good units: {len(df_units)}")
    else:
        print(f"    No unit filter applied.")

    if add_coarse_regions:
        df_units['coarse_region'] = region_parcellation(df_units['region'])

    return df_units


def load_session_spikes(
    unit_filter: str = None,
    remove_duplicate_spikes: bool = True,
) -> pd.DataFrame:
    """
    Load spike times and unit info for analyzable sessions.

    Calls `load_sessions`, restricts `load_units` to those session eids,
    then loads spike times via `load_spikes` and merges session info in.

    Parameters
    ----------
    unit_filter : str or None
        Optional `pandas.DataFrame.query` expression forwarded to
        `load_units`.
    remove_duplicate_spikes : bool
        Forwarded to `load_spikes` as `remove_duplicates`.

    Returns
    -------
    df_spikes : pandas.DataFrame
        Per-unit rows with columns from the units table, a 'spike_times'
        column of numpy arrays, and session info merged on
        ['subject', 'eid'].
    """
    # Load sessions
    df_sessions = load_sessions()  # session info
    eids = df_sessions['eid'].tolist()

    # Load units restricted to analyzable sessions
    df_units = load_units(eids=eids, unit_filter=unit_filter)

    # Load spike times
    print("Loading spike times...")
    df_spiketimes = load_spikes(df_units['uuid'], remove_duplicates=remove_duplicate_spikes)
    # Join spike times with unit info
    df_spikes = df_units.set_index('uuid').join(df_spiketimes).reset_index()
    # Merge session info into spikes dataframe
    df_spikes = pd.merge(
        df_spikes, df_sessions, on=['subject', 'eid'], how='left'
        )

    return df_spikes


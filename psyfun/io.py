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

from one import params
from one.alf.exceptions import ALFObjectNotFound
from brainbox.io.one import SpikeSortingLoader
from iblatlas.atlas import AllenAtlas
atlas = AllenAtlas()

from psyfun.atlas import region_parcellation
from psyfun.config import (ibl_project, paths, df_controls, TASKTIMINGS,
                           PASSIVE_SLOTS)
from psyfun.histology import (list_histology_tifs, insertion_picks,
                              insertion_alignment_uploaded,
                              insertion_alignment_resolved)

PASSIVE_PROTOCOL_TOKEN = 'passive'
SPONTANEOUS_PROTOCOL_TOKEN = 'spontaneous'
PROTOCOL_SPONTANEOUS_DURATION_S = 300.0  # protocol design: 5 min spontaneous
PROTOCOL_REPLAY_DURATION_S = 300.0  # protocol design: 5 min gabor replay
RAW_TASK_COLLECTION_RE = re.compile(r'^raw_task_data_(\d+)$')

# Two-state file-presence vocabulary used by the dataset check.
PRESENT = 'present'
MISSING = 'missing'

# Task-level alf datasets checked per passive slot.
TASK_ALF_FILES = {
    'intervalsTable': '_ibl_passivePeriods.intervalsTable.csv',
    'passiveStims': '_ibl_passiveStims.table.csv',
    'passiveGabor': '_ibl_passiveGabor.table.csv',
}
# Spike sorters whose output may be registered on Alyx, in priority order:
# the first sorter with a registered `spikes.times.npy` wins.
SPIKE_SORTERS = ('iblsorter', 'pykilosort', 'ks2')
# Spike-sorting alf datasets checked per probe (shorthand → dataset filename).
SPIKE_SORTING_FILES = {
    'spikes.times': 'spikes.times.npy',
    'spikes.clusters': 'spikes.clusters.npy',
    'clusters.uuids': 'clusters.uuids.csv',
}
# Bombcell writes this file next to the spike-sorting output on disk.
BOMBCELL_OUTPUT_FILE = 'templates._bc_qMetrics.parquet'


def fetch_sessions(one, save=True):
    """
    Query Alyx for sessions tagged in the psychedelics project and add session
    info to a dataframe. Sessions are restricted to those with the
    passiveChoiceWorld task protocol, dataset-extraction status is checked
    against the Alyx dataset registry, and task protocol timings are added.
    Sessions are sorted and labelled (session_n) by their order.

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
    df_sessions['n_probes'] = df_sessions.apply(lambda x: _count_probes(x['eid'], one), axis='columns')
    df_sessions['n_tasks'] = df_sessions['task_protocol'].apply(lambda x: sum(['passive' in task.lower() for task in x.split('_')]))
    # Check dataset-extraction status against the Alyx dataset registry
    print("Checking datasets...")
    df_sessions = df_sessions.progress_apply(_check_datasets, one=one, axis='columns').copy()
    # Add label for control sessions
    df_sessions['control_recording'] = df_sessions.apply(_label_controls, axis='columns')
    # Fetch task protocol timings and add to dataframe
    print("Fetching protocol timings...")
    df_sessions = df_sessions.progress_apply(_fetch_protocol_timings, one=one, axis='columns').copy()
    # Add LSD administration time
    df_meta = load_metadata()
    df_sessions = df_sessions.apply(_insert_LSD_admin_time, df_metadata=df_meta, axis='columns').copy()
    # Label and sort by session number for each subject
    df_sessions['session_n'] = df_sessions.groupby('subject')['start_time'].rank(method='dense').astype(int)
    df_sessions = df_sessions.sort_values(by=['subject', 'start_time']).reset_index(drop=True)
    # Drop Alyx columns kept only as scratch values during the apply pipeline.
    df_sessions = df_sessions.drop(columns=['projects', 'lab', 'number'], errors='ignore')
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


def _label_controls (session, controls=df_controls):
    eid = session['eid']
    control_session = controls.query('eid == @eid')
    if len(control_session) == 1:
        return True
    elif len(control_session) == 0:
        return False
    elif len(control_session) > 1:
        raise ValueError("More than one entry in df_controls!")


def _count_probes(eid: str, one) -> int:
    """Number of probe insertions registered for `eid`. Warns and returns 0 if none."""
    pids, _ = one.eid2pid(eid)
    if pids is None:
        warnings.warn(f"No probe insertions registered for eid={eid}; setting n_probes=0")
        return 0
    return len(pids)


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
    # Clip to protocol-design duration so all sessions get an equal-length
    # window. One session (c7cf8e25 task_00) has a 600 s alf interval —
    # likely the IBL extractor merged the LSD-filler block into task_00.
    spont_stop = min(float(spont_stop), float(spont_start) + PROTOCOL_SPONTANEOUS_DURATION_S)
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
    Classify one `raw_task_data_NN` and load its FPGA-aligned timings.

    Returns a dict tagged `kind='passive'` (with epoch boundaries; or
    `alf_missing=True` for rig-clock fallback) for passive protocols, or
    `kind='spontaneous'` (rig anchor only) for the LSD-filler protocol.
    Returns None for any other protocol.
    """
    settings = one.load_dataset(eid, '_iblrig_taskSettings.raw.json', raw_col)
    protocol = settings.get('PYBPOD_PROTOCOL', '')
    rig_t0 = _rig_session_datetime(settings)
    if SPONTANEOUS_PROTOCOL_TOKEN in protocol and PASSIVE_PROTOCOL_TOKEN not in protocol:
        return {'kind': 'spontaneous', 'rig_t0': rig_t0}
    if PASSIVE_PROTOCOL_TOKEN not in protocol:
        return None
    alf_col = raw_col.replace('raw_task_data_', 'alf/task_')
    try:
        intervals = one.load_dataset(eid, '_ibl_passivePeriods.intervalsTable.csv', alf_col)
    except ALFObjectNotFound:
        return {'kind': 'passive', 'rig_t0': rig_t0, 'alf_missing': True}
    try:
        gabor = one.load_dataset(eid, '_ibl_passiveGabor.table.csv', alf_col)
    except ALFObjectNotFound:
        gabor = None
    return {'kind': 'passive', **_fpga_timings_from_alf(intervals, gabor), 'rig_t0': rig_t0}


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
        series[col] = np.nan
    runs: list[dict] = []
    for raw_col in _list_raw_task_collections(eid, one):
        run = _load_passive_run(eid, raw_col, one)
        if run is not None:
            runs.append(run)
    passives = [r for r in runs if r['kind'] == 'passive']
    fillers = [r for r in runs if r['kind'] == 'spontaneous']
    anchor = next((p for p in passives if not p.get('alf_missing')), None)
    if anchor is None:
        return series
    for passive_idx, run in enumerate(passives):
        if run.get('alf_missing'):
            run = {**_shift_timings(anchor, anchor['rig_t0'], run['rig_t0']), 'rig_t0': run['rig_t0']}
        slot = PASSIVE_SLOTS[passive_idx]
        for epoch in ('spontaneous', 'rfm', 'replay'):
            for endpoint in ('start', 'stop'):
                series[f'{slot}_{epoch}_{endpoint}'] = run[f'{epoch}_{endpoint}']
    if fillers:
        delta_s = (fillers[0]['rig_t0'] - anchor['rig_t0']).total_seconds()
        series['LSD_admin'] = anchor['spontaneous_start'] + delta_s
    return series


def _insert_LSD_admin_time(series, df_metadata=None):
    assert df_metadata is not None
    # Sessions with a spontaneous-filler protocol already have LSD_admin set
    # by `_fetch_protocol_timings` (start of that protocol in FPGA seconds).
    if pd.notna(series.get('LSD_admin')):
        return series
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


def _raw_task_collections_from_registry(dsr: list[dict]) -> list[str]:
    """`raw_task_data_NN` collection names in the dataset registry, sorted by NN."""
    matches = {
        m.group(0): int(m.group(1))
        for d in dsr
        for m in [RAW_TASK_COLLECTION_RE.match(d['collection'])]
        if m
    }
    return [c for c, _ in sorted(matches.items(), key=lambda kv: kv[1])]


def _list_passive_raw_collections(eid: str, dsr: list[dict], one) -> list[str]:
    """`raw_task_data_NN` collections of the passive runs, in run order.

    Derives the candidate `raw_task_data_NN` collections from the dataset
    registry, then reuses `_load_passive_run`'s protocol classification so
    the spontaneous-only LSD-filler run is excluded.
    """
    passives = []
    for raw_col in _raw_task_collections_from_registry(dsr):
        run = _load_passive_run(eid, raw_col, one)
        if run is not None and run['kind'] == 'passive':
            passives.append(raw_col)
    return passives


def _check_task_alf(present: set, slot: int, raw_col: str | None) -> dict:
    """Status of the three passive-task alf datasets for one passive slot."""
    prefix = PASSIVE_SLOTS[slot]
    if raw_col is None:
        return {f'{prefix}_{short}': MISSING for short in TASK_ALF_FILES}
    alf_col = raw_col.replace('raw_task_data_', 'alf/task_')
    return {
        f'{prefix}_{short}': PRESENT if (alf_col, name) in present else MISSING
        for short, name in TASK_ALF_FILES.items()
    }


def _imec_names(suffix: str, slot: int) -> list[str]:
    """Both IBL SpikeGLX filename forms (`imec0` and `imec00`) for a probe slot."""
    return [
        f'_spikeglx_ephysData_g0_t0.imec{slot}.{suffix}',
        f'_spikeglx_ephysData_g0_t0.imec{slot:02d}.{suffix}',
    ]


def _pick_sorter(dsr: list[dict], probe: str) -> tuple[str, str]:
    """Sorter name and version string of the spike sorting registered for `probe`.

    Walks `SPIKE_SORTERS` in priority order and returns the first sorter
    with a registered `spikes.times.npy`. With a single entry it is taken
    regardless of `default_revision`; with multiple entries the one whose
    `default_revision == 'True'` is preferred (falling back to the first).
    Returns `('', '')` when no sorter output is registered.
    """
    for sorter in SPIKE_SORTERS:
        entries = [
            d for d in dsr
            if d['name'] == 'spikes.times.npy'
            and d['collection'] == f'alf/{probe}/{sorter}'
        ]
        if len(entries) == 1:
            entry = entries[0]
        elif entries:
            entry = next(
                (e for e in entries if e.get('default_revision') == 'True'),
                entries[0],
            )
        else:
            continue
        return sorter, entry.get('version') or ''
    return '', ''


def _check_probe(present: set, dsr: list[dict], slot: int, ins: dict | None,
                 session_path) -> dict:
    """Status of the raw, sync, spike-sorting and bombcell data for one probe slot."""
    prefix = f'probe{slot:02d}'
    if ins is None:
        return {
            f'{prefix}_ap.cbin': MISSING,
            f'{prefix}_sync.npy': MISSING,
            f'{prefix}_sorter': '',
            **{f'{prefix}_{short}': MISSING for short in SPIKE_SORTING_FILES},
            f'{prefix}_spectralDensityLF.npy': MISSING,
            f'{prefix}_bombcell_GOOD': np.nan,
        }
    probe = ins['name']
    raw_col = f'raw_ephys_data/{probe}'
    raw_ap = any((raw_col, name) in present for name in _imec_names('ap.cbin', slot))
    sync = any((raw_col, name) in present for name in _imec_names('sync.npy', slot))
    # Sentinel for the RawEphysQC task: present only when that task completed,
    # flagging probes left with spike-sorting output but no LFP/QC products.
    psd_lf = (raw_col, '_iblqc_ephysSpectralDensityLF.power.npy') in present

    sorter, version = _pick_sorter(dsr, probe)
    sorter_col = f'alf/{probe}/{sorter}'
    spike_status = {
        f'{prefix}_{short}': (
            PRESENT if sorter and (sorter_col, name) in present else MISSING
        )
        for short, name in SPIKE_SORTING_FILES.items()
    }

    bombcell_path = (
        session_path / 'spike_sorters' / sorter / probe
        / 'bombcell' / BOMBCELL_OUTPUT_FILE
    )
    if bombcell_path.is_file():
        bc = pd.read_parquet(bombcell_path)
        good_proportion = (bc['bc_unitType'] == 'GOOD').mean()
    else:
        good_proportion = np.nan
    return {
        f'{prefix}_ap.cbin': PRESENT if raw_ap else MISSING,
        f'{prefix}_sync.npy': PRESENT if sync else MISSING,
        f'{prefix}_sorter': version,
        **spike_status,
        f'{prefix}_spectralDensityLF.npy': PRESENT if psd_lf else MISSING,
        f'{prefix}_bombcell_GOOD': good_proportion,
    }


def _check_histology_probe(eid: str, slot: int, ins: dict | None, one,
                           stacks_present: bool) -> dict:
    """Highest histology-pipeline state reached for one probe slot.

    One of `'resolved'`, `'aligned'`, `'traced'`, `'no-tracing'`,
    `'no-stacks'`, `'no-insertion'`. `'no-insertion'` iff the probe slot has
    no insertion; `'no-stacks'` vs `'no-tracing'` is gated by `stacks_present`.
    """
    key = f'probe{slot:02d}_histology'
    if ins is None:
        return {key: 'no-insertion'}
    full = one.alyx.rest('insertions', 'list', id=ins['id'], no_cache=True)[0]
    if insertion_alignment_resolved(full):
        state = 'resolved'
    elif insertion_alignment_uploaded(full):
        state = 'aligned'
    elif insertion_picks(full):
        state = 'traced'
    elif stacks_present:
        state = 'no-tracing'
    else:
        state = 'no-stacks'
    return {key: state}


def _qc_outcome(extended_qc: dict, key: str) -> str:
    """Extended-QC outcome string for `key`; '' when absent.

    Some entries store a list whose first element is the outcome; others
    store the outcome string directly.
    """
    val = extended_qc.get(key, '')
    if isinstance(val, list):
        return val[0] if val else ''
    return val


def _check_camera(present: set, cam: str, extended_qc: dict) -> dict:
    """Status of raw video and pose tracking plus three video-QC outcomes.

    File-status columns use the raw filename style (`<cam>Camera.raw`,
    `<cam>Camera.lightningPose`). QC columns are emitted under the verbatim
    Alyx `extended_qc` keys with whitespace in the outcome string replaced
    by underscores (e.g. ``'NOT SET'`` → ``'NOT_SET'``).
    """
    raw_video = ('raw_video_data', f'_iblrig_{cam}Camera.raw.mp4') in present
    pose = ('alf', f'_ibl_{cam}Camera.lightningPose.pqt') in present
    capcam = cam.capitalize()
    out = {
        f'{cam}Camera.raw': PRESENT if raw_video else MISSING,
        f'{cam}Camera.lightningPose': PRESENT if pose else MISSING,
    }
    for suffix in ('dropped_frames', 'timestamps', 'pin_state'):
        key = f'_video{capcam}_{suffix}'
        outcome = _qc_outcome(extended_qc, key)
        out[key] = re.sub(r'\s+', '_', outcome) if outcome else ''
    return out


@lru_cache(maxsize=None)
def _check_image_stacks(subject: str, lab: str) -> str:
    """`'present'` iff both an `*_RD.tif` and an `*_GR.tif` stack are published."""
    tifs = list_histology_tifs(subject, lab, params.get())
    has_rd = any(fname.endswith('_RD.tif') for fname in tifs)
    has_gr = any(fname.endswith('_GR.tif') for fname in tifs)
    return PRESENT if has_rd and has_gr else MISSING


def _check_datasets(series, one=None):
    """Add the 37 registry-based check columns for one session.

    See specs/check_session_datasets.md. Dataset presence is read from
    the Alyx `sessions/read` field `data_dataset_session_related`: a
    dataset is present when its `(collection, name)` pair has a non-null
    `data_url`.
    """
    if one is None:
        one = _get_default_connection()
    eid = series['eid']
    # `lab` rides along on the series from sessions/list and is dropped from
    # df_sessions at the end of fetch_sessions; read it locally here.
    lab = series['lab']
    session = one.alyx.rest('sessions', 'read', id=eid)
    dsr = session.get('data_dataset_session_related') or []
    extended_qc = session.get('extended_qc') or {}
    present = {(d['collection'], d['name']) for d in dsr if d['data_url']}
    out = {}
    passives = _list_passive_raw_collections(eid, dsr, one)
    for slot in (0, 1):
        raw_col = passives[slot] if slot < len(passives) else None
        out.update(_check_task_alf(present, slot, raw_col))
    insertions = sorted(
        one.alyx.rest('insertions', 'list', session=eid, no_cache=True),
        key=lambda ins: ins['name'],
    )
    session_path = one.eid2path(eid)
    stacks_present = _check_image_stacks(series['subject'], lab) == PRESENT
    for slot in (0, 1):
        ins = insertions[slot] if slot < len(insertions) else None
        out.update(_check_probe(present, dsr, slot, ins, session_path))
        out.update(_check_histology_probe(eid, slot, ins, one, stacks_present))
    for cam in ('left', 'right', 'body'):
        out.update(_check_camera(present, cam, extended_qc))
    for key, val in out.items():
        series[key] = val
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


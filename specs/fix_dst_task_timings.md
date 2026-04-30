# Fix DST-induced offset and broader misalignment of task-epoch timings

## Goal

Make the `taskNN_*_start/stop` columns in `metadata/sessions.pqt` correctly
identify, in the spike-time clock, the boundaries of each behavioural
sub-epoch (`spontaneous`, `rfm`, `replay`) for every recording in the
dataset, including the 17 BST sessions currently broken by a +3600 s offset
and the sessions where one of the two passive runs was never extracted by
IBL.

**Hard constraint**: do not change how downstream code (`_apply_spike_counts`,
`scripts/population_dimensionality.py`, `scripts/single_unit.py`,
`notebooks/davide/03_brain_states.py`) consumes these columns. The fix lives
inside `_fetch_protocol_timings` and the regenerated `sessions.pqt`.

## What the bug is and how it was verified

### The DST symptom

`psyfun/io.py:_fetch_protocol_timings` computes every `taskNN_*` time as
`(iblrig_event_datetime - alyx_session_start_datetime).seconds`. When the
two datetimes were stored in different timezones (Alyx UTC vs. iblrig local
during BST), the subtraction added +3600 s to every BST session's
timings.

Empirically (5 sessions × ~200 units, sliced spike counts in saved
windows):

```
label                 task00_sp task00_rep  task01_sp task01_rep frac_units_zero_in_t01
BST 2023-03-28           259.1        0.0        0.0        0.0   100.0%
BST 2023-05-16           555.1        0.0        0.0        0.0   100.0%
BST 2023-07-13           183.8        0.0        0.0        0.0   100.0%
GMT 2022-12-15           422.5      510.4      345.7      441.6     48.5%
GMT 2025-03-11          1404.5     1894.3     2142.2     2389.2      0.0%
```

For every BST session every `task01_*` window and every
`task00_replay/rfm` window falls past the end of the spike record.
`task00_spontaneous` (saved `[3600, 3900]`) silently slices the last
~5 minutes of the recording — non-empty, so nothing crashes, but it is
the wrong segment of the recording. The bug never surfaced because empty
windows return all-zero histograms, not errors, and downstream metrics
(PCA / mean rate / fano / LZc) compute on zeros without complaint.

### Why current `_fetch_protocol_timings` no longer reproduces +3600 s

Today, `one.get_details(eid)['start_time']` returns the rig's local clock
(matches iblrig `SESSION_DATETIME` to the microsecond on every sample
checked). So if `_fetch_protocol_timings` were re-run now against the
current Alyx, the +3600 s offset would not reappear — Alyx changed its
behaviour at some point. The offset is baked into the existing
`metadata/sessions.pqt` from a prior extraction. Either way, the fix must
not depend on Alyx's timestamp semantics.

## Beyond DST: structural problems with the current pipeline

Cross-checking saved `taskNN_*` against IBL's FPGA-aligned canonical data
(`alf/task_NN/_ibl_passivePeriods.intervalsTable.csv` and
`_ibl_passiveGabor.table.csv`) shows the current code is wrong by tens of
seconds even in GMT sessions where no DST term applies:

```
session 996f3585 (GMT 2022-12-15):
                   stored          IBL canonical
task00 spont       [0,    300]     [67,    367]
task00 rfm         [300,  681]     [380,   681]
task00 replay      [681,  984]     gabor first→last: [693, 984]
                                   intervalsTable.taskReplay: [693, 4119]
```

Five separate problems:

1. **DST offset** (the original symptom; +3600 s in 17 BST sessions).
2. **Origin offset (~50–80 s in every session)**: the saved
   `task00_spontaneous_start = 0` assumes the iblrig protocol starts at
   FPGA t=0. In reality the experimenter starts ephys first; the protocol
   begins 50–80 s later. Every saved `task00_*` time is shifted earlier by
   that gap. The 5-min spontaneous window in the saved data therefore
   covers FPGA times that are mostly *before* the actual spontaneous
   epoch.
3. **Hard-coded 5-min spontaneous window**: actual `spontaneousActivity`
   intervals from IBL are 300 s here, but other protocol versions could
   differ; the code never checks.
4. **Zero gap between rfm_stop and replay_start**: real intervals show a
   ~12 s gap between RFM end and the first gabor.
5. **Mislabelled task01 in 2025 (3-task) sessions**: protocol structure is
   `[passive, spontaneous, passive]` with the second passive in
   `raw_task_data_02`. The current loop indexes by position in the
   `tasks` list, so `task01_*` columns are populated from the spontaneous
   protocol (`raw_task_data_01`), not the second passive — they point at
   the wrong protocol entirely.
6. **Missed task01 in `n_tasks_list=1` sessions**: 5 sessions have only
   one entry in Alyx's `task_protocol` field but two `raw_task_data_NN/`
   collections on disk. The current pipeline iterates `series['tasks']`
   and never looks at the second collection. `task01_*` ends up NaN and
   the session is dropped by `dropna(subset=TASKTIMINGS)`.

## What "spike-time clock" actually means

Spike times in `alf/probeNN/{spike_sorter}/spikes.times.npy` are in **FPGA
seconds**, with t=0 at the start of the SpikeGLX recording
(`_spikeglx_ephysData_g0_t0`). Bpod and frame2ttl events are mapped onto
this clock by the IBL extractor (`ibllib.io.extractors.ephys_passive`)
via TTL-pulse correlation against the SpikeGLX sync channel
(`_spikeglx_sync.times.npy`). Outputs of that mapping include:

- `_ibl_passivePeriods.intervalsTable.csv` (start/stop of
  `passiveProtocol`, `spontaneousActivity`, `RFM`, `taskReplay`).
- `_ibl_passiveGabor.table.csv` (per-gabor `start`, `stop`, `position`,
  `contrast`, `phase`).
- `_ibl_passiveStims.table.csv` (valve/tone/noise events).

These all share the spike clock by construction. The downstream
`spikes.times.npy` we cache in `data/spikes.h5` (via `fetch_unit_info`,
which calls `one.load_object(eid, collection=probe_collection,
obj='spikes', ...)`) is the same clock. Verified: for BST session
7149e0fc, `passiveGabor.start` ranges (688–984 s for task_00, 3551–3848 s
for task_01) sit cleanly inside the spike record (0–3940 s).

The IBL alignment GUI (`~/code/ibl-alignment-gui`) consumes
`_ibl_passiveGabor.table.csv` directly: `gabor['start']` filtered to
`contrast > 0.1`, split by `position` (35 left / -35 right), PSTHed
against spike times. No further offset is applied. The PSTH itself is
done by `brainbox.task.passive.get_stim_aligned_activity`, which
**bins spikes by depth** (20 µm bins, all units pooled) rather than
per-unit, then z-scores against a 1.0 s pre-stim baseline. The
depth-binned approach is much less noisy than per-unit averaging — every
bin pools spikes from many units.

## When the IBL alf datasets are valid (per-column)

The protocol design is fixed: 5 min spontaneous, 5 min RFM, 5 min
replay, repeated as two passive runs per session. (Some 2025 sessions
have an additional spontaneous-only protocol after LSD as a filler;
those are skipped — they are identified by
`PYBPOD_PROTOCOL == '_iblrig_tasks_spontaneous'` in the
`_iblrig_taskSettings.raw.json` of that `raw_task_data_NN` collection.)

Empirically, durations from `_ibl_passivePeriods.intervalsTable.csv`
across all 40 sessions / 56 alf/task_NN collections:

```
column                  median dur   range            comment
spontaneousActivity     300.0 s      300.0 – 300.1 s  ✓ matches 5 min
RFM                     301.1 s      301.0 – 301.5 s  ✓ matches 5 min (1 s spacer)
taskReplay              ~varies      311 – 3706 s    ✗ does NOT match 5 min
passiveProtocol         ~varies      949 – 4344 s    ✗ does NOT match 15 min
gabor.start range       300 s        296 – 311 s     ✓ matches 5 min replay block
```

One outlier: `c7cf8e25 / alf/task_00` has spontaneousActivity duration
600 s (10 min). All other 55 collections sit at 300 s. The 600 s value
likely reflects the IBL extractor merging the first passive's
spontaneous block with the LSD-filler spontaneous protocol that follows.
Flagged for manual inspection but not handled specially in the fix.

Per-column validity:

| column                                | valid? | use as                          |
|---------------------------------------|--------|---------------------------------|
| `intervalsTable.spontaneousActivity`  | ✓      | `task_NN_spontaneous_start/stop` |
| `intervalsTable.RFM`                  | ✓      | `task_NN_rfm_start/stop`         |
| `intervalsTable.taskReplay[0]`        | ✓      | `task_NN_replay_start` (fallback) |
| `intervalsTable.taskReplay[1]`        | ✗      | unreliable — extends to end of recording |
| `intervalsTable.passiveProtocol`      | ✗      | same problem as taskReplay stop  |
| `passiveGabor.start.min/.max`         | ✓      | `task_NN_replay_start/stop` (preferred) |
| `passiveGabor.stop.max`               | ✓      | last gabor offset (preferred for replay_stop) |

The intervalsTable's `taskReplay` and `passiveProtocol` stop columns
attribute every post-RFM frame2ttl event to the replay block, including
the long quiet period the FPGA keeps recording after the last gabor.
For task_00 that is often ~30 minutes of empty recording. Replay
boundaries must come from `passiveGabor.table.csv` when available, or
from `intervalsTable.taskReplay[0] + 300 s` as a fallback.

### Other useful task datasets (not currently used; flag for future)

- `_ibl_passiveStims.table.csv` — valveOn/Off, toneOn/Off, noiseOn/Off
  events during replay. Not needed for epoch boundaries but useful for
  per-event PSTHs in the validation script.
- `_ibl_passiveRFM.times.npy` — frame-onset times during the RFM block.
  Not needed for epoch boundaries.
- `_iblrig_stimPositionScreen.raw.csv` (rig clock) — used by the
  current code; not in FPGA clock. Useful only as a fallback to compute
  rig-time deltas between protocol starts when alf is missing.

### Sources of truth, in order of preference

1. `alf/task_NN/_ibl_passivePeriods.intervalsTable.csv`
   (`spontaneousActivity`, `RFM` columns; `taskReplay[0]` only) — for
   spontaneous and RFM, FPGA-aligned by IBL.
2. `alf/task_NN/_ibl_passiveGabor.table.csv` (`start.min`,
   `stop.max`) — for replay, FPGA-aligned by IBL.
3. Anchor task's intervalsTable + iblrig wall-clock delta from the
   missing task's `SESSION_DATETIME` — fallback when alf/task_NN is not
   extracted but raw iblrig data exists.
4. NaN — when neither (1) nor (3) is available.

## Coverage of IBL alf extraction across our dataset

```
both task_00 + task_01 (or _02) intervalsTable extracted: 19/40
task_00 only:                                            17/40
neither:                                                  4/40
total sessions missing one or both alf extractions:      21/40
of those, raw_task_data_NN with stim data still on disk: 21/21
```

Every under-extracted session has the raw iblrig data for the missing
task. So a second passive run did happen on disk; IBL just never produced
the alf folder. The 2023 sessions are missing extraction of `alf/task_01`;
the 2025 sessions are missing `alf/task_02` (because their second passive
lives in `raw_task_data_02`, with `raw_task_data_01` being a spontaneous
protocol).

For sessions with no alf, we cannot recover an FPGA-aligned timing
without re-running IBL extraction (which depends on `_spikeglx_sync.*` +
`_iblrig_RFMapStim.raw.bin` + spacer-template correlation). That is out
of scope for this fix. We will derive what we can without re-extraction:

- Where alf for one task exists, use it.
- Where alf is missing for the second task but raw iblrig data is
  present, derive the FPGA-aligned start of that task by adding the
  iblrig wall-clock delta between the two `SESSION_DATETIME`s to the
  FPGA-aligned start of the extracted task. This assumes the rig PC
  clock and FPGA clock advance at the same rate over a ~1 h session,
  which is the same assumption the original (broken) code already made.
  Within the second task, sub-epoch durations are taken from the first
  task's intervalsTable (spontaneous = 300 s, rfm ≈ 301 s) and replay
  bounds come from the second task's
  `_iblrig_stimPositionScreen.raw.csv` gabor times, shifted by the same
  wall-clock-to-FPGA offset.
- Where neither alf is available (4 sessions): leave as NaN and drop.
  Flag for re-extraction as future work.

## Fix design

`_fetch_protocol_timings` becomes a per-task lookup against alf-canonical
data, with a documented fallback for the second passive when its alf
folder is missing. Pseudocode:

```python
def _fetch_protocol_timings(series, one):
    eid = series['eid']

    # 1. enumerate the actual passive task collections, ignoring any
    #    spontaneous-only protocol (raw_task_data_NN matching
    #    'spontaneous' is skipped).
    passive_collections = _list_passive_task_collections(eid, one)

    # 2. For each passive collection, pull canonical timings from alf if
    #    present.
    extracted = {}
    for n, raw_col in enumerate(passive_collections):
        alf_col = _alf_col_for(raw_col)  # mapping verified by inspection
        try:
            intervals = one.load_dataset(eid, '_ibl_passivePeriods.intervalsTable.csv', alf_col)
            gabor     = one.load_dataset(eid, '_ibl_passiveGabor.table.csv', alf_col)
            extracted[n] = {
                'spontaneous': (intervals['spontaneousActivity'].iloc[0],
                                intervals['spontaneousActivity'].iloc[1]),
                'rfm':         (intervals['RFM'].iloc[0],
                                intervals['RFM'].iloc[1]),
                'replay':      (float(gabor['start'].min()),
                                float(gabor['start'].max())),
                'rig_t0':      datetime.fromisoformat(_rig_session_datetime(eid, raw_col, one)),
            }
        except ALFObjectNotFound:
            extracted[n] = None

    # 3. For collections with no alf, derive from rig wall-clock delta.
    anchor = next((v for v in extracted.values() if v is not None), None)
    if anchor is None:
        return series  # no canonical data; everything stays NaN
    for n, raw_col in enumerate(passive_collections):
        if extracted[n] is not None:
            continue
        rig_dt = datetime.fromisoformat(_rig_session_datetime(eid, raw_col, one))
        rig_offset = (rig_dt - anchor['rig_t0']).total_seconds()
        # Replay bounds from this task's iblrig gabor csv, in rig
        # seconds, then translated to FPGA via rig_offset and the
        # anchor's known FPGA-rig offset.
        # Spontaneous + RFM duration assumed identical to anchor; their
        # FPGA-time anchors derived from anchor's offsets-from-rig-t0.
        ...

    # 4. Write into series with the existing column names.
    for n, vals in extracted.items():
        if vals is None: continue
        series[f'task{n:02d}_spontaneous_start'] = vals['spontaneous'][0]
        series[f'task{n:02d}_spontaneous_stop']  = vals['spontaneous'][1]
        series[f'task{n:02d}_rfm_start']         = vals['rfm'][0]
        series[f'task{n:02d}_rfm_stop']          = vals['rfm'][1]
        series[f'task{n:02d}_replay_start']      = vals['replay'][0]
        series[f'task{n:02d}_replay_stop']       = vals['replay'][1]
    return series
```

(Pseudocode only — to be tightened in implementation. The exact mapping
between `raw_task_data_NN` and `alf/task_NN` for 3-task sessions still
needs verification: in 2025 sessions raw is
`[raw_task_data_00 passive, raw_task_data_01 spontaneous, raw_task_data_02 passive]`
and we need to confirm whether the alf side numbers passive-only
collections or numbers all collections.)

`LSD_admin` continues to be set by `_insert_LSD_admin_time` from
`metadata/metadata.csv`. The existing column names in `TASKTIMINGS`
remain unchanged — downstream consumers see the same interface, just
with corrected values.

## Validation

The remaining empirical question is whether the timings the *fixed*
pipeline writes into `sessions.pqt` produce visible neural responses
when used as PSTH anchors against the spike times we load from
`data/spikes.h5`. Validation script
`scripts/validate_gabor_alignment.py` does this end-to-end **using only
local-pipeline data and a hand-rolled PSTH** (no `brainbox` PSTH
helpers). The point is to test our consumption path, not IBL's.

Per session:

1. Take prospective replay-window onsets from the proposed pipeline
   output for that session (and, before the pipeline change, also
   directly from `_ibl_passiveGabor.table.csv` as a control). Filter to
   `contrast > 0.1`, split by `position` (35 / -35).
2. Load all spike times for the session from `data/spikes.h5` (per
   uuid), and unit depths from `data/units.pqt`.
3. Build a depth-binned spike-count matrix (20 µm × 10 ms) over the full
   recording, pooling all units.
4. For each gabor onset, extract a `[-0.4, +1.0] s` window per depth
   bin. Z-score per depth bin against a 1.0 s pre-stim baseline,
   averaged across trials.
5. Plot `[depth × time]` z-score heatmaps per (probe, task collection,
   side ∈ {left, right}).

A vertical band of color around `t = 0+` confirms the timings align with
the spike data.

### Notes from earlier validation attempts

- A first attempt averaged raw firing rates across all units in a
  session (~3000 units, no quality filter) and produced washed-out
  PSTHs with no visible response. Most "units" in `units.pqt` are MUA
  or barely-firing junk (75 % have firing rate < 1.25 Hz; 50 % have
  presence_ratio < 0.05).
- A second attempt computed per-unit z-scored PSTHs and sorted units by
  post-stim peak |z|. Heatmaps still looked noisy because per-unit z is
  high-variance with only 60 trials per (side × contrast cell).
- The depth-binned approach (above) is what the IBL alignment GUI uses
  and what makes responses visible in non-V1 regions. Replicating its
  logic by hand (not by importing it) is the right move.
- Filtering gabors to `contrast > 0.1` (excluding 0.0 and 0.0625) is
  necessary; including catch trials dilutes the response.
- A 0.4 s baseline gives unstable per-unit z-scores; 1.0 s is the IBL
  default and what we should use.

### Validation results (depth-binned, hand-rolled)

`scripts/validate_gabor_alignment.py` (depth bin 20 µm, time bin 10 ms,
1 s pre-stim baseline, z-scored per depth bin) produces these
statistics per (session, alf/task collection, side):

```
session            collection    side  n   resp_bins  peak_z  peak_t
BST 2023-03-28     alf/task_00   L    60   20.0 %     15.05    +75 ms
                   alf/task_00   R    60   14.3 %      8.13     +5 ms
                   alf/task_01   L    60   21.6 %     21.29    +45 ms
                   alf/task_01   R    60   15.5 %      9.95   +125 ms
BST 2023-05-16     alf/task_00   L    60   15.6 %      7.75   +245 ms
                   alf/task_00   R    60   17.1 %      9.27   +175 ms
GMT 2022-12-15     alf/task_01   L    60   16.2 %      7.00    +25 ms
                   alf/task_01   R    60   18.8 %      5.70   +265 ms
GMT 2025-03-11     alf/task_00   L    59    7.9 %      5.56     +5 ms
                   alf/task_00   R    60    8.4 %      4.04   +275 ms
```

`resp_bins` = fraction of depth bins with peak |z| > 3 in [0, 0.3] s
post-stim. `peak_z` = max |z| across depths in that window.
`peak_t` = time of that peak relative to gabor onset.

Two facts that nail down the alignment:

1. Every peak time sits in [5, 275] ms post-stim, the physiological
   visual-response window. Random misalignment would put peaks
   anywhere in [-1000, +1000] ms.
2. BST 7149e0fc / task_01 currently has 100 % of units with zero
   spikes in the saved (buggy) windows. Same spike data, same gabor
   times, but anchored on FPGA-aligned `passiveGabor.start`, it has
   the strongest response of the four sessions tested
   (peak_z = 21.29 at +45 ms, 22 % of depth bins responsive). The
   spike data is correctly aligned; only the saved anchors were wrong.

### Done when

- The validation plot shows a stim-locked response (visible vertical
  red/blue band at t ≳ 0) for at least one BST session and at least one
  GMT session, in both `task_00` and `task_01` (or `task_02` for 2025
  sessions).
- A regeneration check confirms `sessions.pqt` `task_NN_*_start/stop`
  columns satisfy `0 <= value <= spike_max + 1` for every retained
  session, in both BST and GMT.
- No change to `TASKTIMINGS` column names; downstream scripts run
  unchanged on the regenerated parquet.

## Out of scope (flag separately)

- Re-running IBL extraction (`ephys_passive`) for the 4 sessions with no
  alf and the 17 sessions with only one alf. Doable but heavy; needs
  `_spikeglx_sync.*` and the spacer template.
- Refactoring the consumer interface to accept `(t0, t1)` pairs directly
  instead of stringly-typed `taskNN_epoch_start/stop` columns.
- Re-running `data/popdim_*.pqt` and `data/singleunit_*.pqt` after
  `sessions.pqt` is regenerated.

## Affected files

- `psyfun/io.py`: rewrite `_fetch_protocol_timings`.
- `tests/test_fetch_protocol_timings.py` (new): unit tests for the
  alf-canonical and alf-fallback branches.
- `scripts/validate_gabor_alignment.py` (new): empirical validation via
  gabor-onset PSTH on local spike data.
- `scripts/validate_dst_fix.py` (new, one-off): regeneration sanity
  checks (BST sessions decrease by 3600 s; GMT sessions match within
  1 s; previously-NaN `task01_*` columns become non-NaN where raw data
  exists; every value sits in spike-record range).
- `metadata/sessions.pqt`: regenerated (separate commit).

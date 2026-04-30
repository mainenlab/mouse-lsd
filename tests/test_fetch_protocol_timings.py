"""Tests for psyfun.io._fetch_protocol_timings and its pure-logic helpers."""
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from one.alf.exceptions import ALFObjectNotFound

from psyfun import io


def _intervals_df(spont, rfm, replay) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "passiveProtocol": [0.0, replay[1] + 1],
            "spontaneousActivity": list(spont),
            "RFM": list(rfm),
            "taskReplay": list(replay),
        },
        index=["start", "stop"],
    )
    df.index.name = ""
    return df.reset_index()


def _gabor_df(starts, stops) -> pd.DataFrame:
    return pd.DataFrame({"start": starts, "stop": stops})


class FakeOne:
    """Minimal ONE stub. Datasets keyed by (eid, filename, collection)."""

    def __init__(self, datasets: dict, list_datasets_map: dict):
        self._datasets = datasets
        self._list_map = list_datasets_map

    def load_dataset(self, eid, name, collection):
        key = (eid, name, collection)
        if key not in self._datasets:
            raise ALFObjectNotFound(name)
        return self._datasets[key]

    def list_datasets(self, eid):
        return self._list_map[eid]


def test_canonical_alf_for_two_passive_runs():
    """Both passive runs have alf data: read everything from canonical files."""
    eid = "EID1"
    intervals_00 = _intervals_df((50, 350), (362, 663), (675, 2872))
    gabor_00 = _gabor_df([685.0, 985.0], [685.3, 985.3])
    intervals_01 = _intervals_df((2916, 3216), (3228, 3529), (3541, 3941))
    gabor_01 = _gabor_df([3551.0, 3848.0], [3551.3, 3848.3])
    settings_00 = {"PYBPOD_PROTOCOL": "_iblrig_tasks_passiveChoiceWorld",
                   "SESSION_DATETIME": "2023-03-28T11:00:00"}
    settings_01 = {"PYBPOD_PROTOCOL": "_iblrig_tasks_passiveChoiceWorld",
                   "SESSION_DATETIME": "2023-03-28T11:50:00"}
    one = FakeOne(
        datasets={
            (eid, "_iblrig_taskSettings.raw.json", "raw_task_data_00"): settings_00,
            (eid, "_iblrig_taskSettings.raw.json", "raw_task_data_01"): settings_01,
            (eid, "_ibl_passivePeriods.intervalsTable.csv", "alf/task_00"): intervals_00,
            (eid, "_ibl_passivePeriods.intervalsTable.csv", "alf/task_01"): intervals_01,
            (eid, "_ibl_passiveGabor.table.csv", "alf/task_00"): gabor_00,
            (eid, "_ibl_passiveGabor.table.csv", "alf/task_01"): gabor_01,
        },
        list_datasets_map={
            eid: [
                "raw_task_data_00/_iblrig_taskSettings.raw.json",
                "raw_task_data_01/_iblrig_taskSettings.raw.json",
                "alf/task_00/_ibl_passivePeriods.intervalsTable.csv",
                "alf/task_01/_ibl_passivePeriods.intervalsTable.csv",
                "alf/task_00/_ibl_passiveGabor.table.csv",
                "alf/task_01/_ibl_passiveGabor.table.csv",
            ],
        },
    )
    series = pd.Series({"eid": eid})
    result = io._fetch_protocol_timings(series, one=one)
    assert result["task00_spontaneous_start"] == 50
    assert result["task00_spontaneous_stop"] == 350
    assert result["task00_rfm_start"] == 362
    assert result["task00_rfm_stop"] == 663
    assert result["task00_replay_start"] == 685.0
    assert result["task00_replay_stop"] == 985.3
    assert result["task01_spontaneous_start"] == 2916
    assert result["task01_replay_start"] == 3551.0
    assert result["task01_replay_stop"] == 3848.3


def test_skip_spontaneous_filler_in_3_task_session():
    """raw_task_data_01 is spontaneous filler; second passive lives in _02."""
    eid = "EID2"
    intervals_00 = _intervals_df((50, 350), (362, 663), (675, 980))
    gabor_00 = _gabor_df([685.0, 985.0], [685.3, 985.3])
    settings_00 = {"PYBPOD_PROTOCOL": "_iblrig_tasks_passiveChoiceWorld",
                   "SESSION_DATETIME": "2025-03-11T18:00:00"}
    settings_01_filler = {"PYBPOD_PROTOCOL": "_iblrig_tasks_spontaneous",
                          "SESSION_DATETIME": "2025-03-11T18:20:00"}
    settings_02 = {"PYBPOD_PROTOCOL": "_iblrig_tasks_passiveChoiceWorld",
                   "SESSION_DATETIME": "2025-03-11T18:50:00"}
    one = FakeOne(
        datasets={
            (eid, "_iblrig_taskSettings.raw.json", "raw_task_data_00"): settings_00,
            (eid, "_iblrig_taskSettings.raw.json", "raw_task_data_01"): settings_01_filler,
            (eid, "_iblrig_taskSettings.raw.json", "raw_task_data_02"): settings_02,
            (eid, "_ibl_passivePeriods.intervalsTable.csv", "alf/task_00"): intervals_00,
            (eid, "_ibl_passiveGabor.table.csv", "alf/task_00"): gabor_00,
            # No alf for task_02: must fall back via rig-clock delta.
        },
        list_datasets_map={
            eid: [
                "raw_task_data_00/_iblrig_taskSettings.raw.json",
                "raw_task_data_01/_iblrig_taskSettings.raw.json",
                "raw_task_data_02/_iblrig_taskSettings.raw.json",
                "alf/task_00/_ibl_passivePeriods.intervalsTable.csv",
                "alf/task_00/_ibl_passiveGabor.table.csv",
            ],
        },
    )
    series = pd.Series({"eid": eid})
    result = io._fetch_protocol_timings(series, one=one)
    # Filler must NOT populate task01_*; second passive should land in task01_*.
    assert result["task00_spontaneous_start"] == 50
    rig_delta_s = (datetime.fromisoformat("2025-03-11T18:50:00")
                   - datetime.fromisoformat("2025-03-11T18:00:00")).total_seconds()
    assert result["task01_spontaneous_start"] == pytest.approx(50 + rig_delta_s)
    assert result["task01_replay_start"] == pytest.approx(685.0 + rig_delta_s)


def test_fallback_when_alf_missing_for_second_passive():
    """2-task session, alf only for task_00. task_01 derived via rig delta."""
    eid = "EID3"
    intervals_00 = _intervals_df((50, 350), (362, 663), (675, 980))
    gabor_00 = _gabor_df([685.0, 985.0], [685.3, 985.3])
    settings_00 = {"PYBPOD_PROTOCOL": "_iblrig_tasks_passiveChoiceWorld",
                   "SESSION_DATETIME": "2023-05-16T11:00:00"}
    settings_01 = {"PYBPOD_PROTOCOL": "_iblrig_tasks_passiveChoiceWorld",
                   "SESSION_DATETIME": "2023-05-16T11:50:00"}
    one = FakeOne(
        datasets={
            (eid, "_iblrig_taskSettings.raw.json", "raw_task_data_00"): settings_00,
            (eid, "_iblrig_taskSettings.raw.json", "raw_task_data_01"): settings_01,
            (eid, "_ibl_passivePeriods.intervalsTable.csv", "alf/task_00"): intervals_00,
            (eid, "_ibl_passiveGabor.table.csv", "alf/task_00"): gabor_00,
        },
        list_datasets_map={
            eid: [
                "raw_task_data_00/_iblrig_taskSettings.raw.json",
                "raw_task_data_01/_iblrig_taskSettings.raw.json",
                "alf/task_00/_ibl_passivePeriods.intervalsTable.csv",
                "alf/task_00/_ibl_passiveGabor.table.csv",
            ],
        },
    )
    series = pd.Series({"eid": eid})
    result = io._fetch_protocol_timings(series, one=one)
    rig_delta = 50 * 60.0
    assert result["task01_spontaneous_start"] == pytest.approx(50 + rig_delta)
    assert result["task01_rfm_start"] == pytest.approx(362 + rig_delta)
    assert result["task01_replay_stop"] == pytest.approx(985.3 + rig_delta)


def test_no_alf_anywhere_returns_nan():
    """No canonical data at all: leave timing fields untouched (NaN-safe)."""
    eid = "EID4"
    settings_00 = {"PYBPOD_PROTOCOL": "_iblrig_tasks_passiveChoiceWorld",
                   "SESSION_DATETIME": "2024-01-01T10:00:00"}
    one = FakeOne(
        datasets={
            (eid, "_iblrig_taskSettings.raw.json", "raw_task_data_00"): settings_00,
        },
        list_datasets_map={
            eid: ["raw_task_data_00/_iblrig_taskSettings.raw.json"],
        },
    )
    series = pd.Series({"eid": eid})
    result = io._fetch_protocol_timings(series, one=one)
    assert "task00_spontaneous_start" not in result or pd.isna(
        result.get("task00_spontaneous_start")
    )


def test_replay_fallback_when_gabor_missing():
    """alf intervalsTable present, passiveGabor missing: replay = taskReplay[0] + 300."""
    eid = "EID5"
    intervals_00 = _intervals_df((50, 350), (362, 663), (675, 4000))
    settings_00 = {"PYBPOD_PROTOCOL": "_iblrig_tasks_passiveChoiceWorld",
                   "SESSION_DATETIME": "2023-04-01T10:00:00"}
    one = FakeOne(
        datasets={
            (eid, "_iblrig_taskSettings.raw.json", "raw_task_data_00"): settings_00,
            (eid, "_ibl_passivePeriods.intervalsTable.csv", "alf/task_00"): intervals_00,
        },
        list_datasets_map={
            eid: [
                "raw_task_data_00/_iblrig_taskSettings.raw.json",
                "alf/task_00/_ibl_passivePeriods.intervalsTable.csv",
            ],
        },
    )
    series = pd.Series({"eid": eid})
    result = io._fetch_protocol_timings(series, one=one)
    assert result["task00_replay_start"] == 675
    assert result["task00_replay_stop"] == pytest.approx(675 + 300)

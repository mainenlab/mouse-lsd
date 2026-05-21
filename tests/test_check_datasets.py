"""Tests for psyfun.io._check_datasets and its per-modality helpers."""
import math

import pandas as pd
import pytest

from one.alf.exceptions import ALFObjectNotFound

from psyfun import io


def dsr_entry(name, collection, data_url="https://example/url",
              default_revision="True", revision="", version=""):
    """A `data_dataset_session_related` entry with the keys the check reads."""
    return {
        "name": name, "collection": collection, "data_url": data_url,
        "default_revision": default_revision, "revision": revision,
        "version": version,
    }


class FakeAlyx:
    """Minimal Alyx REST stub. `rest_map` is keyed by (endpoint, action)."""

    def __init__(self, rest_map: dict):
        self._rest_map = rest_map

    def rest(self, endpoint, action, **kwargs):
        handler = self._rest_map.get((endpoint, action), [])
        return handler(kwargs) if callable(handler) else handler


class FakeOne:
    """ONE stub. `datasets` keyed by (eid, filename, collection) for load_dataset."""

    def __init__(self, datasets=None, rest_map=None, path_map=None):
        self._datasets = datasets or {}
        self.alyx = FakeAlyx(rest_map or {})
        self._path_map = path_map or {}

    def load_dataset(self, eid, name, collection, revision=None):
        key = (eid, name, collection)
        if key not in self._datasets:
            raise ALFObjectNotFound(name)
        return self._datasets[key]

    def eid2path(self, eid):
        return self._path_map[eid]


# --- _raw_task_collections_from_registry ----------------------------------

def test_raw_task_collections_from_registry_sorted():
    dsr = [
        dsr_entry("x", "raw_task_data_02"),
        dsr_entry("y", "alf"),
        dsr_entry("z", "raw_task_data_00"),
    ]
    assert io._raw_task_collections_from_registry(dsr) == [
        "raw_task_data_00", "raw_task_data_02"
    ]


# --- _list_passive_raw_collections ----------------------------------------

def test_list_passive_raw_collections_skips_filler():
    eid = "E"
    dsr = [
        dsr_entry("_iblrig_taskSettings.raw.json", "raw_task_data_00"),
        dsr_entry("_iblrig_taskSettings.raw.json", "raw_task_data_01"),
        dsr_entry("_iblrig_taskSettings.raw.json", "raw_task_data_02"),
    ]
    one = FakeOne(datasets={
        (eid, "_iblrig_taskSettings.raw.json", "raw_task_data_00"):
            {"PYBPOD_PROTOCOL": "_iblrig_tasks_passiveChoiceWorld",
             "SESSION_DATETIME": "2025-03-11T18:00:00"},
        (eid, "_iblrig_taskSettings.raw.json", "raw_task_data_01"):
            {"PYBPOD_PROTOCOL": "_iblrig_tasks_spontaneous",
             "SESSION_DATETIME": "2025-03-11T18:20:00"},
        (eid, "_iblrig_taskSettings.raw.json", "raw_task_data_02"):
            {"PYBPOD_PROTOCOL": "_iblrig_tasks_passiveChoiceWorld",
             "SESSION_DATETIME": "2025-03-11T18:50:00"},
    })
    assert io._list_passive_raw_collections(eid, dsr, one) == [
        "raw_task_data_00", "raw_task_data_02"
    ]


# --- _check_task_alf -------------------------------------------------------

def test_check_task_alf_present_and_missing():
    present = {("alf/task_00", "_ibl_passivePeriods.intervalsTable.csv")}
    out = io._check_task_alf(present, 0, "raw_task_data_00")
    assert out["pre_intervalsTable"] == io.PRESENT
    assert out["pre_passiveStims"] == io.MISSING
    assert out["pre_passiveGabor"] == io.MISSING


def test_check_task_alf_no_slot():
    out = io._check_task_alf(set(), 1, None)
    assert set(out) == {
        "post_intervalsTable", "post_passiveStims", "post_passiveGabor"
    }
    assert all(v == io.MISSING for v in out.values())


# --- _pick_sorter ---------------------------------------------------------

def test_pick_sorter_priority_order():
    dsr = [
        dsr_entry("spikes.times.npy", "alf/probe00/pykilosort",
                  version="pykilosort_ibl_1.4.1"),
        dsr_entry("spikes.times.npy", "alf/probe00/iblsorter",
                  version="iblsorter_1.9.0"),
    ]
    assert io._pick_sorter(dsr, "probe00") == ("iblsorter", "iblsorter_1.9.0")


def test_pick_sorter_none_registered():
    assert io._pick_sorter([], "probe00") == ("", "")


def test_pick_sorter_prefers_default_revision():
    dsr = [
        dsr_entry("spikes.times.npy", "alf/probe00/pykilosort",
                  default_revision="False", revision="2024-01-01"),
        dsr_entry("spikes.times.npy", "alf/probe00/pykilosort",
                  default_revision="True", revision="2025-06-01",
                  version="pykilosort_ibl_1.4.1"),
    ]
    assert io._pick_sorter(dsr, "probe00") == (
        "pykilosort", "pykilosort_ibl_1.4.1"
    )


def test_pick_sorter_single_non_default_revision():
    dsr = [
        dsr_entry("spikes.times.npy", "alf/probe00/pykilosort",
                  default_revision="False", revision="2024-01-01",
                  version="pykilosort_ibl_1.4.1"),
    ]
    assert io._pick_sorter(dsr, "probe00") == (
        "pykilosort", "pykilosort_ibl_1.4.1"
    )


# --- _check_probe ----------------------------------------------------------

def test_check_probe_complete(tmp_path):
    ins = {"name": "probe00", "id": "PID0"}
    bombcell = (tmp_path / "spike_sorters" / "iblsorter" / "probe00"
                / "bombcell" / io.BOMBCELL_OUTPUT_FILE)
    bombcell.parent.mkdir(parents=True)
    pd.DataFrame({"bc_unitType": ["GOOD", "MUA", "NOISE"]}).to_parquet(bombcell)
    present = {
        ("raw_ephys_data/probe00", "_spikeglx_ephysData_g0_t0.imec0.ap.cbin"),
        ("raw_ephys_data/probe00", "_spikeglx_ephysData_g0_t0.imec0.sync.npy"),
        ("alf/probe00/iblsorter", "spikes.times.npy"),
        ("alf/probe00/iblsorter", "spikes.clusters.npy"),
        ("alf/probe00/iblsorter", "clusters.uuids.csv"),
        ("raw_ephys_data/probe00", "_iblqc_ephysSpectralDensityLF.power.npy"),
    }
    dsr = [dsr_entry("spikes.times.npy", "alf/probe00/iblsorter",
                     version="iblsorter_1.9.0")]
    out = io._check_probe(present, dsr, 0, ins, tmp_path)
    assert out["probe00_ap.cbin"] == io.PRESENT
    assert out["probe00_sync.npy"] == io.PRESENT
    assert out["probe00_spectralDensityLF.npy"] == io.PRESENT
    assert out["probe00_sorter"] == "iblsorter_1.9.0"
    assert out["probe00_spikes.times"] == io.PRESENT
    assert out["probe00_spikes.clusters"] == io.PRESENT
    assert out["probe00_clusters.uuids"] == io.PRESENT
    assert out["probe00_bombcell_GOOD"] == 1 / 3
    assert "probe00_bombcell" not in out


def test_check_probe_imec_double_digit_form(tmp_path):
    ins = {"name": "probe01", "id": "PID1"}
    present = {
        ("raw_ephys_data/probe01", "_spikeglx_ephysData_g0_t0.imec01.ap.cbin"),
    }
    out = io._check_probe(present, [], 1, ins, tmp_path)
    assert out["probe01_ap.cbin"] == io.PRESENT
    # raw present, derived files absent
    assert out["probe01_sync.npy"] == io.MISSING
    assert out["probe01_sorter"] == ""
    assert out["probe01_spikes.times"] == io.MISSING
    assert out["probe01_spikes.clusters"] == io.MISSING
    assert out["probe01_clusters.uuids"] == io.MISSING
    assert out["probe01_spectralDensityLF.npy"] == io.MISSING
    assert math.isnan(out["probe01_bombcell_GOOD"])


def test_check_probe_raw_missing(tmp_path):
    ins = {"name": "probe00", "id": "PID0"}
    out = io._check_probe(set(), [], 0, ins, tmp_path)
    assert out["probe00_ap.cbin"] == io.MISSING
    assert out["probe00_sync.npy"] == io.MISSING
    assert out["probe00_spikes.times"] == io.MISSING
    assert out["probe00_spikes.clusters"] == io.MISSING
    assert out["probe00_clusters.uuids"] == io.MISSING
    assert math.isnan(out["probe00_bombcell_GOOD"])


def test_check_probe_no_insertion():
    out = io._check_probe(set(), [], 1, None, None)
    assert out["probe01_ap.cbin"] == io.MISSING
    assert out["probe01_sync.npy"] == io.MISSING
    assert out["probe01_spikes.times"] == io.MISSING
    assert out["probe01_spikes.clusters"] == io.MISSING
    assert out["probe01_clusters.uuids"] == io.MISSING
    assert math.isnan(out["probe01_bombcell_GOOD"])
    assert out["probe01_sorter"] == ""
    assert out["probe01_spectralDensityLF.npy"] == io.MISSING


def test_check_probe_sorter_is_version_verbatim(tmp_path):
    ins = {"name": "probe00", "id": "PID0"}
    present = {
        ("raw_ephys_data/probe00", "_spikeglx_ephysData_g0_t0.imec0.ap.cbin"),
        ("alf/probe00/iblsorter", "spikes.times.npy"),
        ("alf/probe00/iblsorter", "spikes.clusters.npy"),
        ("alf/probe00/iblsorter", "clusters.uuids.csv"),
    }
    dsr = [dsr_entry("spikes.times.npy", "alf/probe00/iblsorter",
                     default_revision="False",
                     revision="2025-06-01", version="iblsorter_1.9.0")]
    out = io._check_probe(present, dsr, 0, ins, tmp_path)
    assert out["probe00_sorter"] == "iblsorter_1.9.0"


# --- _check_camera ---------------------------------------------------------

def test_check_camera_qc_and_status():
    present = {("raw_video_data", "_iblrig_leftCamera.raw.mp4")}
    extended_qc = {
        "_videoLeft_dropped_frames": ["PASS", 14, 0],
        "_videoLeft_timestamps": "WARNING",
        # no pin_state entry
    }
    out = io._check_camera(present, "left", extended_qc)
    assert out["leftCamera.raw"] == io.PRESENT
    assert out["leftCamera.lightningPose"] == io.MISSING
    assert out["_videoLeft_dropped_frames"] == "PASS"
    assert out["_videoLeft_timestamps"] == "WARNING"
    assert out["_videoLeft_pin_state"] == ""


def test_check_camera_timestamps_whitespace_normalised():
    out = io._check_camera(set(), "left", {"_videoLeft_timestamps": "NOT SET"})
    assert out["_videoLeft_timestamps"] == "NOT_SET"


def test_check_camera_raw_missing():
    out = io._check_camera(set(), "body", {})
    assert out["bodyCamera.raw"] == io.MISSING
    assert out["bodyCamera.lightningPose"] == io.MISSING
    assert out["_videoBody_dropped_frames"] == ""
    assert out["_videoBody_timestamps"] == ""
    assert out["_videoBody_pin_state"] == ""


def test_check_camera_keys_union_across_cameras():
    keys = set()
    for cam in ("left", "right", "body"):
        keys.update(io._check_camera(set(), cam, {}).keys())
    expected = set()
    for cam in ("left", "right", "body"):
        capcam = cam.capitalize()
        expected.update({f"{cam}Camera.raw", f"{cam}Camera.lightningPose"})
        expected.update({
            f"_video{capcam}_dropped_frames",
            f"_video{capcam}_timestamps",
            f"_video{capcam}_pin_state",
        })
    assert keys == expected
    assert len(expected) == 15


# --- _check_histology_probe ------------------------------------------------

def _hist_one(full: dict) -> "FakeOne":
    return FakeOne(rest_map={("insertions", "list"): [full]})


def test_check_histology_probe_no_insertion():
    out = io._check_histology_probe("E", 1, None, FakeOne(), False)
    assert out == {"probe01_histology": "no-insertion"}


def test_check_histology_probe_no_stacks_when_picks_empty_and_no_stacks():
    ins = {"name": "probe00", "id": "PID0"}
    full = {"id": "PID0", "json": {"xyz_picks": [], "extended_qc": {}}}
    out = io._check_histology_probe("E", 0, ins, _hist_one(full), False)
    assert out == {"probe00_histology": "no-stacks"}


def test_check_histology_probe_no_tracing_when_xyz_picks_empty():
    ins = {"name": "probe00", "id": "PID0"}
    full = {"id": "PID0", "json": {"xyz_picks": [], "extended_qc": {}}}
    out = io._check_histology_probe("E", 0, ins, _hist_one(full), True)
    assert out == {"probe00_histology": "no-tracing"}


def test_check_histology_probe_traced_when_picks_only():
    ins = {"name": "probe00", "id": "PID0"}
    full = {"id": "PID0", "json": {"xyz_picks": [[1, 2, 3]], "extended_qc": {}}}
    out = io._check_histology_probe("E", 0, ins, _hist_one(full), True)
    assert out == {"probe00_histology": "traced"}


def test_check_histology_probe_aligned_when_alignment_count_only():
    ins = {"name": "probe00", "id": "PID0"}
    full = {
        "id": "PID0",
        "json": {
            "xyz_picks": [[1, 2, 3]],
            "extended_qc": {"alignment_count": 2},
        },
    }
    out = io._check_histology_probe("E", 0, ins, _hist_one(full), True)
    assert out == {"probe00_histology": "aligned"}


def test_check_histology_probe_resolved_wins():
    ins = {"name": "probe00", "id": "PID0"}
    full = {
        "id": "PID0",
        "json": {
            "xyz_picks": [[1, 2, 3]],
            "extended_qc": {"alignment_count": 2, "alignment_resolved": True},
        },
    }
    out = io._check_histology_probe("E", 0, ins, _hist_one(full), True)
    assert out == {"probe00_histology": "resolved"}


# --- _check_datasets -------------------------------------------------------

def test_check_datasets_data_url_none_is_absent(tmp_path, monkeypatch):
    """A registry entry with a null `data_url` does not count as present."""
    eid = "E"
    session = {
        "extended_qc": {},
        "data_dataset_session_related": [
            dsr_entry("_ibl_passivePeriods.intervalsTable.csv", "alf/task_00"),
            dsr_entry("_ibl_passiveStims.table.csv", "alf/task_00", data_url=None),
            dsr_entry("_iblrig_taskSettings.raw.json", "raw_task_data_00"),
        ],
    }
    one = FakeOne(
        datasets={
            (eid, "_iblrig_taskSettings.raw.json", "raw_task_data_00"):
                {"PYBPOD_PROTOCOL": "_iblrig_tasks_passiveChoiceWorld",
                 "SESSION_DATETIME": "2025-03-11T18:00:00"},
        },
        rest_map={
            ("sessions", "read"): session,
            ("insertions", "list"): [],
        },
        path_map={eid: tmp_path},
    )
    monkeypatch.setattr(io, "_check_image_stacks", lambda subject, lab: io.MISSING)
    series = pd.Series({"eid": eid, "subject": "S", "lab": "L"})
    out = io._check_datasets(series, one=one)
    assert out["pre_intervalsTable"] == io.PRESENT
    assert out["pre_passiveStims"] == io.MISSING
    assert out["pre_passiveGabor"] == io.MISSING
    assert out["post_intervalsTable"] == io.MISSING
    assert "image_stacks" not in out


# --- _check_image_stacks ---------------------------------------------------

def _patch_tifs(monkeypatch, tifs):
    io._check_image_stacks.cache_clear()
    monkeypatch.setattr(io, "list_histology_tifs", lambda subject, lab, par: tifs)


def test_image_stacks_present_when_rd_and_gr(monkeypatch):
    _patch_tifs(monkeypatch, ["x_RD.tif", "y_GR.tif"])
    assert io._check_image_stacks("S", "L") == io.PRESENT


def test_image_stacks_missing_when_only_one_suffix(monkeypatch):
    _patch_tifs(monkeypatch, ["x_RD.tif", "y_RD.tif"])
    assert io._check_image_stacks("S", "L") == io.MISSING


def test_image_stacks_missing_when_empty(monkeypatch):
    _patch_tifs(monkeypatch, [])
    assert io._check_image_stacks("S", "L") == io.MISSING


# --- fetch_sessions --------------------------------------------------------

def test_fetch_sessions_drops_unused_alyx_columns(monkeypatch):
    """`projects`, `lab`, `number`, `tasks` are absent from the final df."""
    sessions_list = [{
        "id": "E",
        "subject": "S",
        "start_time": "2025-03-11T18:00:00",
        "url": "https://example/E",
        "task_protocol": "_iblrig_tasks_passiveChoiceWorld/_iblrig_tasks_passiveChoiceWorld",
        "projects": ["psychedelics"],
        "lab": "L",
        "number": 1,
    }]
    one = FakeOne(rest_map={("sessions", "list"): sessions_list})
    monkeypatch.setattr(io, "_count_probes", lambda eid, one: 0)
    monkeypatch.setattr(io, "_check_datasets", lambda s, one: s)
    monkeypatch.setattr(io, "_fetch_protocol_timings", lambda s, one: s)
    monkeypatch.setattr(io, "load_metadata", lambda: pd.DataFrame())
    monkeypatch.setattr(io, "_insert_LSD_admin_time", lambda s, df_metadata: s)
    df = io.fetch_sessions(one, save=False)
    assert not {"projects", "lab", "number", "tasks"} & set(df.columns)


def test_check_datasets_docstring_references_new_spec():
    """Module-level spec reference points to `specs/check_session_datasets.md`."""
    doc = io._check_datasets.__doc__
    assert "specs/check_session_datasets.md" in doc
    assert "specs/check_dataset_extraction.md" not in doc

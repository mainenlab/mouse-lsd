"""
Validate `metadata/sessions.pqt` task-epoch timings against local spike data.

Two checks per session, against the timings saved in `paths['sessions']`:

1. Per-epoch spike-coverage: counts spikes in each saved
   `task_NN_<epoch>_<start..stop>` window across all units. An epoch
   with zero total spikes means the window sits outside the recording
   (the bug this fix addresses).

2. Gabor-onset PSTH around the saved replay window: loads FPGA-aligned
   gabor onsets from `_ibl_passiveGabor.table.csv`, keeps those that
   fall inside the saved `[replay_start, replay_stop]` window
   (filter to `contrast > 0.1`), and computes a depth-binned z-scored
   PSTH on local spike data. If the saved window is correct, every
   gabor onset falls inside, and a stim-locked response is visible
   (peak |z| > 3 within 0–300 ms post-stim).

Usage:
    python scripts/validate_gabor_alignment.py            # stats only
    python scripts/validate_gabor_alignment.py --plot     # save heatmaps
    python scripts/validate_gabor_alignment.py --plot --show
"""

import argparse
import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from one.api import ONE
from one.alf.exceptions import ALFObjectNotFound

from psyfun.config import paths


SESSIONS = [
    ("BST 2023-03-28", "7149e0fc-a52d-4e93-849c-edc22d54e7a5"),
    ("BST 2023-05-16", "cbc72b2f-2906-497e-8df0-dfaf825ffb39"),
    ("GMT 2022-12-15", "996f3585-b804-4a3d-878a-1c15d708962b"),
    ("GMT 2025-03-11", "c7cf8e25-1e2c-4b03-a5f5-5a049f1cd228"),
]
EPOCHS = ("spontaneous", "rfm", "replay")
TASKS = ("task00", "task01")
PRE = 1.0
POST = 1.0
T_BIN = 0.01
D_BIN = 20.0
Z_THRESH = 3.0
RESPONSE_WINDOW = (0.0, 0.3)


def session_spikes_by_depth(
    df_units: pd.DataFrame, eid: str, h5: h5py.File,
) -> tuple[np.ndarray, np.ndarray]:
    units = df_units[df_units["eid"] == eid]
    times_chunks, depths_chunks = [], []
    for _, row in units.iterrows():
        if row["uuid"] not in h5 or pd.isna(row["depth"]):
            continue
        ts = h5[row["uuid"]]["times"][:]
        if len(ts) == 0:
            continue
        times_chunks.append(ts)
        depths_chunks.append(np.full(len(ts), row["depth"], dtype=np.float32))
    if not times_chunks:
        return np.array([]), np.array([])
    return np.concatenate(times_chunks), np.concatenate(depths_chunks)


def bin_spikes_2d(
    spike_times: np.ndarray, spike_depths: np.ndarray,
    t_edges: np.ndarray, d_edges: np.ndarray,
) -> np.ndarray:
    counts, _, _ = np.histogram2d(spike_depths, spike_times, bins=[d_edges, t_edges])
    return counts


def zscored_psth(
    binned: np.ndarray, t_edges: np.ndarray, events: np.ndarray,
    pre: float, post: float, t_bin: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_pre, n_post = int(pre / t_bin), int(post / t_bin)
    n_total = n_pre + n_post
    valid = (events - pre >= t_edges[0]) & (events + post <= t_edges[-1])
    events = events[valid]
    centers = np.arange(-n_pre, n_post) * t_bin + t_bin / 2
    if len(events) == 0:
        return centers, np.full((binned.shape[0], n_total), np.nan)
    idx_event = np.searchsorted(t_edges, events) - 1
    n_depth = binned.shape[0]
    stim_trials = np.zeros((n_depth, n_total, len(events)))
    base_trials = np.zeros((n_depth, n_pre, len(events)))
    for i, idx in enumerate(idx_event):
        stim_trials[:, :, i] = binned[:, idx - n_pre:idx + n_post]
        base_trials[:, :, i] = binned[:, idx - n_pre:idx]
    avg_stim = stim_trials.mean(axis=2)
    avg_base = base_trials.mean(axis=2)
    base_mean = avg_base.mean(axis=1, keepdims=True)
    base_std = avg_base.std(axis=1, keepdims=True)
    base_std[base_std == 0] = np.nan
    z = (avg_stim - base_mean) / base_std
    return centers, z


def response_stats(z: np.ndarray, centers: np.ndarray, window: tuple[float, float], thresh: float):
    mask = (centers >= window[0]) & (centers < window[1])
    z_post = z[:, mask]
    finite = np.isfinite(z_post).all(axis=1)
    if not finite.any():
        return dict(n_depth=0, n_responsive=0, frac=0.0, peak_z=np.nan, peak_t=np.nan)
    z_finite = z_post[finite]
    peak_per_bin = np.abs(z_finite).max(axis=1)
    n_responsive = int((peak_per_bin > thresh).sum())
    peak_idx = int(peak_per_bin.argmax())
    peak_t_idx = int(np.abs(z_finite[peak_idx]).argmax())
    return dict(
        n_depth=int(finite.sum()),
        n_responsive=n_responsive,
        frac=n_responsive / int(finite.sum()),
        peak_z=float(peak_per_bin.max()),
        peak_t=float(centers[mask][peak_t_idx]),
    )


def epoch_spike_count(spike_times: np.ndarray, t0: float, t1: float) -> int:
    return int(((spike_times >= t0) & (spike_times <= t1)).sum())


def gabor_events_in_window(
    one: ONE, eid: str, t0: float, t1: float,
) -> tuple[np.ndarray | None, int]:
    """
    Search alf/task_NN/_ibl_passiveGabor.table.csv collections and return
    high-contrast gabor onsets falling inside `[t0, t1]`. The collection
    is matched by overlap (gabor times must fall in the window). Returns
    (None, 0) if no alf gabor data is found for any collection.
    """
    for alf_col in (f"alf/task_{i:02d}" for i in range(3)):
        try:
            gabor = one.load_dataset(eid, "_ibl_passiveGabor.table.csv", alf_col)
        except ALFObjectNotFound:
            continue
        starts = gabor.loc[gabor["contrast"] > 0.1, "start"].to_numpy(dtype=np.float64)
        in_win = (starts >= t0) & (starts <= t1)
        if in_win.any():
            return starts[in_win], len(starts)
    return None, 0


def coverage_check(
    sessions_row: pd.Series, spike_times: np.ndarray,
) -> dict:
    """Print per-epoch spike counts; return summary counts."""
    summary = dict(epochs_total=0, epochs_zero=0, epochs_nan=0)
    print("  saved-window spike coverage:")
    for task in TASKS:
        for epoch in EPOCHS:
            summary["epochs_total"] += 1
            t0 = sessions_row.get(f"{task}_{epoch}_start", np.nan)
            t1 = sessions_row.get(f"{task}_{epoch}_stop", np.nan)
            if pd.isna(t0) or pd.isna(t1):
                summary["epochs_nan"] += 1
                print(f"    {task}_{epoch:11s}: NaN window")
                continue
            n = epoch_spike_count(spike_times, t0, t1)
            if n == 0:
                summary["epochs_zero"] += 1
            print(f"    {task}_{epoch:11s}: [{t0:7.1f}, {t1:7.1f}]  spikes={n:>9d}  "
                  f"{'EMPTY' if n == 0 else ''}")
    return summary


def gabor_psth_check(
    sessions_row: pd.Series, eid: str, one: ONE, binned: np.ndarray, t_edges: np.ndarray,
) -> list[tuple]:
    panels = []
    print("  gabor-onset PSTH against saved replay window:")
    for task in TASKS:
        t0 = sessions_row.get(f"{task}_replay_start", np.nan)
        t1 = sessions_row.get(f"{task}_replay_stop", np.nan)
        if pd.isna(t0) or pd.isna(t1):
            print(f"    {task}: replay window NaN; skipping PSTH")
            continue
        events, total = gabor_events_in_window(one, eid, t0, t1)
        if events is None:
            print(f"    {task}: no alf passiveGabor matches saved window")
            continue
        centers, z = zscored_psth(binned, t_edges, events, PRE, POST, T_BIN)
        if not np.isfinite(z).any():
            print(f"    {task}: PSTH unstable")
            continue
        stats = response_stats(z, centers, RESPONSE_WINDOW, Z_THRESH)
        print(
            f"    {task}: events_in_window={len(events):3d}/{total:3d}  "
            f"depth_bins={stats['n_depth']:3d}  "
            f"resp(|z|>{Z_THRESH:.0f}@[{RESPONSE_WINDOW[0]:.1f},{RESPONSE_WINDOW[1]:.1f}]s)="
            f"{stats['n_responsive']:3d} ({100*stats['frac']:5.1f}%)  "
            f"peak_z={stats['peak_z']:5.2f}  peak_t={1000*stats['peak_t']:+.0f}ms"
        )
        panels.append((task, len(events), centers, z))
    return panels


def make_figure(label: str, eid: str, panels: list, d_edges: np.ndarray) -> plt.Figure:
    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 5), squeeze=False)
    last_im = None
    for j, (task, n_events, centers, z) in enumerate(panels):
        ax = axes[0, j]
        last_im = ax.imshow(
            z, aspect="auto", cmap="bwr", vmin=-5, vmax=5, origin="lower",
            extent=[centers[0], centers[-1], d_edges[0], d_edges[-1]],
        )
        ax.axvline(0, ls="--", color="k", lw=0.8)
        ax.set_title(f"{task}\nn={n_events}", fontsize=10)
        ax.set_xlabel("Time from gabor onset (s)")
        if j == 0:
            ax.set_ylabel("Depth on probe (µm)")
    if last_im is not None:
        fig.colorbar(last_im, ax=axes[0, -1], label="z-score", fraction=0.05, pad=0.02)
    fig.suptitle(f"{label} ({eid[:8]})", fontsize=11)
    fig.tight_layout()
    return fig


def analyze_session(
    label: str, eid: str, sessions_row: pd.Series,
    one: ONE, df_units: pd.DataFrame, h5: h5py.File, plot: bool,
) -> tuple[dict, plt.Figure | None]:
    spike_times, spike_depths = session_spikes_by_depth(df_units, eid, h5)
    if len(spike_times) == 0:
        return dict(epochs_total=0, epochs_zero=0, epochs_nan=0), None
    print(f"\n=== {label} ({eid[:8]}) ===")
    summary = coverage_check(sessions_row, spike_times)
    t_max = spike_times.max() + T_BIN
    t_edges = np.arange(0, t_max, T_BIN)
    d_edges = np.arange(0, np.ceil(spike_depths.max() / D_BIN) * D_BIN + D_BIN, D_BIN)
    binned = bin_spikes_2d(spike_times, spike_depths, t_edges, d_edges)
    panels = gabor_psth_check(sessions_row, eid, one, binned, t_edges)
    fig = make_figure(label, eid, panels, d_edges) if (plot and panels) else None
    return summary, fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    one = ONE()
    df_sessions = pd.read_parquet(paths["sessions"])
    df_units = pd.read_parquet(paths["units"])
    out_dir = paths["figures"]
    out_dir.mkdir(parents=True, exist_ok=True)

    totals = dict(epochs_total=0, epochs_zero=0, epochs_nan=0)
    figs = []
    with h5py.File(paths["spikes"], "r") as h5:
        for label, eid in SESSIONS:
            row = df_sessions[df_sessions["eid"] == eid]
            if row.empty:
                print(f"{label}: eid not in sessions.pqt; skipping")
                continue
            summary, fig = analyze_session(
                label, eid, row.iloc[0], one, df_units, h5, args.plot,
            )
            for k in totals:
                totals[k] += summary[k]
            if fig is not None:
                slug = label.replace(" ", "_")
                out = out_dir / f"validate_gabor_alignment_{slug}.png"
                fig.savefig(out, dpi=120)
                print(f"  saved {out}")
                figs.append(fig)

    print("\n=== overall ===")
    print(f"  epochs evaluated:    {totals['epochs_total']}")
    print(f"  epochs with NaN:     {totals['epochs_nan']:3d}  "
          f"({100*totals['epochs_nan']/max(totals['epochs_total'],1):5.1f}%)")
    print(f"  epochs with 0 spikes: {totals['epochs_zero']:3d}  "
          f"({100*totals['epochs_zero']/max(totals['epochs_total'],1):5.1f}%)")
    if args.show and figs:
        plt.show()


if __name__ == "__main__":
    main()

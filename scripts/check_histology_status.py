"""
Check Alyx for the histology-pipeline status of every probe in every valid
mouse-lsd session, write a per-pid status table, and print a prioritized
to-do list grouped by the next required action.

Four checks per probe insertion:

1. ``image_stacks``     – ≥2 ``.tif`` files listed under
   ``{HTTP_DATA_SERVER}/histology/{lab}/{subject}/downsampledStacks_25/sample2ARA/``
   (the same listing ``ibl_alignment_gui.loaders.histology_loader.download_histology_data``
   uses). Per-subject; we cache a single probe per subject.
2. ``picks``            – ``len(insertion.json.xyz_picks) > 0``.
3. ``alignment_uploaded`` – ``insertion.json.extended_qc.alignment_count > 0``.
4. ``alignment_resolved`` – ``insertion.json.extended_qc.alignment_resolved is True``.

Valid sessions come from ``psyfun.io.load_sessions``. Output is written to
``metadata/histology_status.pqt``. No large data is downloaded — only Alyx
REST calls and a directory-listing HTTP GET per subject.
"""
import argparse
import re

import pandas as pd
import requests
from tqdm import tqdm

from one import params
from one.api import ONE

from psyfun.config import paths
from psyfun.io import load_sessions


HIST_REL = 'histology/{lab}/{subject}/downsampledStacks_25/sample2ARA/'

TODO_TASKS = [
    ('image_stacks', 'TRACE STACKS'),
    ('picks', 'TRACE PICKS'),
    ('alignment_uploaded', 'ALIGN'),
    ('alignment_resolved', 'RESOLVE'),
]


def list_histology_tifs(subject: str, lab: str, par) -> list[str]:
    """Return the list of ``.tif`` filenames published for ``subject`` in ``lab``.

    Empty list if the directory does not exist or the request fails.
    """
    url = f'{par.HTTP_DATA_SERVER}/' + HIST_REL.format(lab=lab, subject=subject)
    try:
        r = requests.get(
            url,
            auth=(par.HTTP_DATA_SERVER_LOGIN, par.HTTP_DATA_SERVER_PWD),
            timeout=15,
        )
    except requests.RequestException:
        return []
    if r.status_code != 200:
        return []
    return [m + '.tif' for m in re.findall(r'href="(.*).tif"', r.text)]


def insertion_picks(insertion: dict) -> bool:
    return bool((insertion.get('json') or {}).get('xyz_picks') or [])


def insertion_alignment_uploaded(insertion: dict) -> bool:
    eq = ((insertion.get('json') or {}).get('extended_qc') or {})
    n = eq.get('alignment_count')
    return bool(n) and n > 0


def insertion_alignment_resolved(insertion: dict) -> bool:
    eq = ((insertion.get('json') or {}).get('extended_qc') or {})
    return eq.get('alignment_resolved') is True


def status_rows(one: ONE, df_sessions: pd.DataFrame) -> list[dict]:
    """Iterate sessions → pids and collect a status row per pid."""
    par = params.get()
    image_stacks_cache: dict[tuple[str, str], bool] = {}
    rows: list[dict] = []

    for _, sess in tqdm(
        df_sessions.iterrows(), total=len(df_sessions), desc='sessions'
    ):
        eid = sess['eid']
        subject = sess['subject']
        lab = sess['lab']

        cache_key = (subject, lab)
        if cache_key not in image_stacks_cache:
            tifs = list_histology_tifs(subject, lab, par)
            image_stacks_cache[cache_key] = len(tifs) >= 2

        try:
            pids, probe_names = one.eid2pid(eid)
        except Exception as e:
            print(f'  [warn] eid2pid failed for {eid}: {e}')
            pids, probe_names = [], []

        if not pids:
            rows.append({
                'subject': subject,
                'eid': eid,
                'start_time': sess['start_time'],
                'pid': None,
                'probe': None,
                'image_stacks': image_stacks_cache[cache_key],
                'picks': False,
                'alignment_uploaded': False,
                'alignment_resolved': False,
            })
            continue

        for pid, probe in zip(pids, probe_names):
            pid = str(pid)
            try:
                ins = one.alyx.rest(
                    'insertions', 'list', id=pid, no_cache=True
                )[0]
            except Exception as e:
                print(f'  [warn] insertion fetch failed for {pid}: {e}')
                ins = {'json': {}}
            rows.append({
                'subject': subject,
                'eid': eid,
                'start_time': sess['start_time'],
                'pid': pid,
                'probe': probe,
                'image_stacks': image_stacks_cache[cache_key],
                'picks': insertion_picks(ins),
                'alignment_uploaded': insertion_alignment_uploaded(ins),
                'alignment_resolved': insertion_alignment_resolved(ins),
            })

    return rows


def next_task(row: pd.Series) -> str | None:
    """Return the first incomplete pipeline step for a probe, or None if done."""
    for col, label in TODO_TASKS:
        if not row[col]:
            return label
    return None


def print_todo_list(df_status: pd.DataFrame) -> None:
    """Print a prioritized to-do list grouped by next required action."""
    df = df_status.copy()
    df['task'] = df.apply(next_task, axis='columns')
    df = df.dropna(subset=['task'])
    df['date'] = pd.to_datetime(df['start_time'], format='ISO8601').dt.strftime('%Y-%m-%d')

    for _, label in TODO_TASKS:
        block = df[df['task'] == label].sort_values(['subject', 'date'])
        if block.empty:
            continue
        print(f'\n== {label} ({len(block)}) ==')
        for _, r in block.iterrows():
            probe = r['probe'] if r['probe'] else '-'
            print(f'{label} - {r["subject"]} - {r["date"]} - {probe}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--limit', type=int, default=None,
        help='Process only the first N valid sessions (for verification).',
    )
    parser.add_argument(
        '--out',
        default=str(paths['sessions'].parent / 'histology_status.pqt'),
        help='Output parquet path.',
    )
    args = parser.parse_args()

    one = ONE(base_url='https://alyx.internationalbrainlab.org')

    df_sessions = load_sessions(drop_extra_columns=False)
    if args.limit:
        df_sessions = df_sessions.head(args.limit)
        print(f'Limiting to first {args.limit}')

    rows = status_rows(one, df_sessions)
    df_status = pd.DataFrame(rows)

    counts = df_status[
        ['image_stacks', 'picks', 'alignment_uploaded', 'alignment_resolved']
    ].sum()
    print('\ncounts (True):')
    print(counts.to_string())

    print_todo_list(df_status)

    df_status.to_parquet(args.out, index=False)
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()

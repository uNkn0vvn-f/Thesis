"""Thematic funds against a nonclaimant AI-revenue benchmark, on the selected labels.

Runs src.analysis.flows.alignment, pointed at the fund-quarter exposure table and the
claim ladder of THESIS_LABELS, with its working files in
data/interim/analysis/alignment_<labels>/. Estimates go to TABLES/alignment_flows.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src import paths as tp  # noqa: E402
from src.analysis.robustness.common import TABLES, INTERIM, LABELS, path  # noqa: E402


def main():
    from src.analysis.flows import alignment as W
    work = INTERIM / 'analysis' / ('alignment_' + LABELS)
    for d in ('out', 'logs', 'tables'):
        (work / d).mkdir(parents=True, exist_ok=True)
    W.EXP, W.LAD = str(path('exposure')), str(path('ladder'))
    W.OUT, W.LOGS = str(work / 'out'), str(work / 'logs')
    tp.RESULTS = work                 # alignment writes RESULTS/tables/alignment_flows.csv
    W.main()
    T = pd.read_parquet(work / 'out' / 'alignment.parquet')
    T.insert(0, 'labels', LABELS)
    T.to_csv(TABLES / 'alignment_flows.csv', index=False, float_format='%.6g')
    f = T[(T.test == 'flows') & (T.cut == 'all claimants')].iloc[0]
    print('below minus above the benchmark, flows %.3f (SE %.3f), 95%% interval %.2f to %.2f'
          % (f['diff'], f.se_diff, f['diff'] - 1.96 * f.se_diff, f['diff'] + 1.96 * f.se_diff))


if __name__ == '__main__':
    main()

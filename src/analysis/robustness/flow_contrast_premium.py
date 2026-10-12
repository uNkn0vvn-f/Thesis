"""The flow-premium regressions (src.analysis.flows.claim_rungs) on the selected labels.

Runs claim_rungs unchanged, pointed at the claim ladder of THESIS_LABELS, with its working
files in data/interim/analysis/claim_rungs_<labels>/. Writes
TABLES/flow_contrast_premium.csv and flow_contrast_paths.csv, the inputs of
src/analysis/plot_flow_contrast.py.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src import paths as tp  # noqa: E402
from src.analysis.robustness.common import TABLES, INTERIM, LABELS, path  # noqa: E402


def main():
    from src.analysis.flows import claim_rungs as F
    work = INTERIM / 'analysis' / ('claim_rungs_' + LABELS)
    for d in ('out', 'logs', 'tables'):
        (work / d).mkdir(parents=True, exist_ok=True)
    F.LAD = str(path('ladder'))
    F.OUT, F.LOGS = str(work / 'out'), str(work / 'logs')
    tp.RESULTS = work                 # claim_rungs writes RESULTS/tables/flow_contrast_premium.csv
    F.main()
    for name in ('flow_contrast_premium.csv', 'flow_contrast_paths.csv'):
        shutil.copyfile(work / 'tables' / name, TABLES / name)
    T = pd.read_csv(TABLES / 'flow_contrast_premium.csv')
    r = T[(T.spec == 'sponsor x month') & (T.rung == -25)].iloc[0]
    print('theme with support minus process-only %.3f (SE %.3f)' % (-r.coef, r.se))


if __name__ == '__main__':
    main()

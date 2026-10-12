"""The age control of the flow contrast: what the 'age' variable is, and two alternatives.

The control is log(1 + months since the fund's first month in the N-PORT flow panel).
N-PORT monthly flows begin in late 2019, so for funds that existed before then the variable
counts months since panel entry, not since inception. A fund is treated as left-censored if
its first flow month is no later than 2020-03 (the first estimation month); for the others
months since first N-PORT flow is close to age since launch. Inception dates are not in the
local data, so no inception-based control is estimated.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.analysis.robustness.flow_contrast_sensitivity import LADDER, panel, fit, row, RN, XC, INTERIM, TABLES  # noqa: E402


def main():
    """Describe the age variable and re-estimate the flow contrast under each age
    specification."""
    D = panel()
    A0 = pd.read_parquet(LADDER, columns=['series_id', 'month'])
    born = pd.to_datetime(A0.groupby('series_id').month.min())
    D['first_month'] = D.series_id.map(born)
    D['left_censored'] = D.first_month <= pd.Timestamp('2020-03-01')
    # How many fund-months of each claim group start at the beginning of the panel (left-
    # censored).
    log = {'first_month_distribution_funds': born.dt.to_period('Q').astype(str).value_counts().sort_index().head(8).to_dict(),
           'share_fund_months_left_censored': float(D.left_censored.mean()),
           'r5_share_fund_months_left_censored': float(D.loc[D.r5 == 1, 'left_censored'].mean()),
           'r2_share_fund_months_left_censored': float(D.loc[D.r2 == 1, 'left_censored'].mean()),
           'r5_funds_launched_in_panel': int(D.loc[(D.r5 == 1) & ~D.left_censored, 'series_id'].nunique()),
           'r5_funds': int(D.loc[D.r5 == 1, 'series_id'].nunique()),
           'r2_funds_launched_in_panel': int(D.loc[(D.r2 == 1) & ~D.left_censored, 'series_id'].nunique()),
           'r2_funds': int(D.loc[D.r2 == 1, 'series_id'].nunique())}
    # The baseline, no age control, age bins for funds launched within the panel, and two
    # sample restrictions.
    rows = []
    xn = RN + XC
    b, V, s = fit(D, xn)
    rows.append(row('baseline (log 1 + months since panel entry)', b, V['series_id'], s, xn))
    xs = RN + ['lsize', 'ret', 'nf_lag_w']
    b, V, s = fit(D, xs)
    rows.append(row('no age control', b, V['series_id'], s, xs))
    bins = pd.cut(D.age_m, [-1, 6, 12, 24, 36, 60, 1000], labels=False)
    D['lc'] = D.left_censored.astype(float)
    for k in range(1, 6):
        D['ab%d' % k] = ((bins == k) & ~D.left_censored).astype(float)
    xa = RN + ['lsize', 'ret', 'nf_lag_w', 'lc'] + ['ab%d' % k for k in range(1, 6)]
    b, V, s = fit(D, xa)
    rows.append(row('age bins for funds launched in panel + left-censored indicator', b, V['series_id'], s, xa))
    new = D[~D.left_censored]
    b, V, s = fit(new, xn)
    rows.append(row('funds launched within the panel only', b, V['series_id'], s, xn))
    b, V, s = fit(D[D.age_m >= 12], xn)
    rows.append(row('at least 12 months in panel', b, V['series_id'], s, xn))
    R = pd.DataFrame(rows)
    R.to_csv(TABLES / 'flow_contrast_age.csv', index=False, float_format='%.6g')
    (TABLES / 'flow_contrast_age_log.json').write_text(json.dumps(log, indent=2), encoding='utf-8')
    print(R[['spec', 'r5_minus_r2', 'se', 'ci_lo', 'ci_hi', 'n', 'funds', 'r5_funds', 'r2_funds']].round(3).to_string())
    print(json.dumps(log, indent=2))


if __name__ == '__main__':
    main()

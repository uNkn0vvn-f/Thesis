"""Checks on the monthly flow panel behind the flow contrast, the alignment comparison and the event analysis.

1. Month alignment. N-PORT reports three monthly returns per filing (MONTH_1..3). The
   panel maps MONTH_1 to the oldest month. If that is right, index-fund returns line up
   with the market return of the same calendar month and not with a shifted month.
2. Flow definition. Previous quarter-end net assets are rolled forward through three
   months of returns and flows and compared with the observed quarter end (an outside
   anchor: neither endpoint is built from the tested flow). Candidate definitions:
   sales - |redemptions| (used), sales + reinvestment - |redemptions|, no flow at all.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.analysis.robustness.common import TABLES, INTERIM, RAW  # noqa: E402


def main():
    F = pd.read_parquet(INTERIM / 'analysis' / 'fund_flows' / 'out' / 'fund_month_flows.parquet')
    F = F[F.series_id.notna() & (F.series_id != '')].copy()
    out = {}

    # ---- 1. alignment against the market factor ----
    ff = pd.read_parquet(RAW / 'wrds' / 'ff_daily.parquet')
    ff.columns = [c.lower() for c in ff.columns]
    dcol = [c for c in ff.columns if 'date' in c][0]
    ff[dcol] = pd.to_datetime(ff[dcol])
    mk = ff.assign(m=ff[dcol].dt.to_period('M'))
    mkt = (1 + mk.mktrf + mk.rf).groupby(mk.m).prod() - 1
    mkt = 100 * mkt
    names = pd.read_parquet(INTERIM / 'holdings' / 'holdings_bridged.parquet',
                            columns=['series_id', 'series_name']).drop_duplicates('series_id')
    idx = names[names.series_name.str.contains(r'S&P 500 Index|Total Stock Market Index|500 Index Fund',
                                                case=False, regex=True)].series_id
    R = F[F.series_id.isin(idx) & F.ret.notna()][['series_id', 'month', 'ret']].copy()
    R['m'] = R.month.dt.to_period('M')
    res = {}
    for shift in (-2, -1, 0, 1, 2):
        x = R.assign(mm=R.m + shift).merge(mkt.rename('mkt'), left_on='mm', right_index=True)
        res[shift] = {'corr': float(np.corrcoef(x.ret, x.mkt)[0, 1]),
                      'median_abs_diff_pp': float((x.ret - x.mkt).abs().median()), 'n': len(x)}
    out['alignment_index_funds'] = {'funds': int(R.series_id.nunique()), 'by_shift_months': res}

    # ---- 2. identity: roll previous quarter-end assets forward ----
    F = F.sort_values(['series_id', 'month'])
    F['r'] = F.ret / 100
    F['reinv'] = F.reinv.fillna(0).abs()
    q = F.groupby(['series_id', 'period'])
    Q = q.agg(na_end=('na_qend', 'first'), n=('month', 'size'), m0=('month', 'min')).reset_index()
    Q = Q[Q.n == 3]
    months = F.set_index(['series_id', 'period'])
    defs = {'sales - |redeem| (used)': F.sales.abs() - F.redeem.abs(),
            'sales + reinv - |redeem|': F.sales.abs() + F.reinv - F.redeem.abs(),
            'sales - redeem (signed as filed)': F.sales - F.redeem,
            'no flow': pd.Series(0.0, index=F.index)}
    Q = Q.sort_values(['series_id', 'period'])
    Q['na_prev'] = Q.groupby('series_id').na_end.shift()
    Q['gap'] = (Q.period - Q.groupby('series_id').period.shift()).dt.days
    Q = Q[Q.na_prev.notna() & Q.gap.between(85, 95) & (Q.na_prev > 1e6)]
    keyset = set(zip(Q.series_id, Q.period))
    sub = F[[k in keyset for k in zip(F.series_id, F.period)]].copy()
    ident = {}
    for name, flow in defs.items():
        sub['f'] = flow.loc[sub.index].values
        sub = sub.sort_values(['series_id', 'period', 'month'])
        roll = []
        for (s, p), g in sub.groupby(['series_id', 'period'], sort=False):
            roll.append((s, p, g.r.values, g.f.values))
        prev = dict(zip(zip(Q.series_id, Q.period), Q.na_prev))
        end = dict(zip(zip(Q.series_id, Q.period), Q.na_end))
        resid = []
        for s, p, r, f in roll:
            na = prev[(s, p)]
            if np.isnan(r).any():
                continue
            for k in range(3):
                na = na * (1 + r[k]) + f[k]
            resid.append(abs(na / end[(s, p)] - 1))
        resid = np.array(resid)
        resid = resid[np.isfinite(resid)]          # quarter-ends reported as zero assets
        ident[name] = {'median_abs_resid': float(np.median(resid)),
                       'share_within_1pct': float(np.mean(resid < 0.01)), 'n': len(resid)}
    out['identity_quarter_roll_forward'] = ident
    (TABLES / 'flow_panel_checks.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()

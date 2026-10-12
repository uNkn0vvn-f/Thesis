"""Portfolio exposure per fund-quarter, and the claim that was public at that date.

Two things this does:

1. CONTEMPORANEOUS ALIGNMENT. A fund's claim is taken from the last prospectus that was
   PUBLIC on the holdings report date, not from whether it ever made the claim. A check
   showed what `ever` does to anything with a date in it: 585 ever-positive theme
   funds against 68 as of ChatGPT. Comparing a 2026 claim with a 2021 portfolio would
   repeat that error inside this test.

2. COVERED-SLEEVE RENORMALISATION. Babina reaches 64.7% of portfolio value and Eisfeldt
   75.3%, because both are built from US employment data. Averaging over the whole
   portfolio with zeros for the uncovered part would understate every fund, and would
   understate global funds most. Each measure is averaged over the part of the portfolio
   where it is observed, and the covered share travels with the number.

Expected exposure is the size-decile x sector bucket mean from src/holdings/b01_stock_panel.py, so a fund
that simply owns the biggest technology names scores zero abnormal exposure. That
benchmark is US-only, because Compustat Global has no shares outstanding; the raw share
is global and is reported alongside.
"""
import io, os

import numpy as np
import pandas as pd
from src import paths as tp  # noqa: E402

from src.fund_claims.asof import expire, MAX_AGE_DAYS                        # noqa: E402

OUT = tp.out('holdings')
LOGS = tp.logs('holdings')
os.makedirs(OUT, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)

HOLD = tp.out('holdings', 'holdings_bridged.parquet')
BENCH = tp.out('holdings')
PANEL = tp.out('fund_claims', 'panel_fund_filing.parquet')

MEAS = ['pct_developer', 'pct_enabler', 'pct_adopter', 'pct_beneficiary',
        'has_developer', 'has_enabler', 'intense_adopter']
OBS = {'pct_developer': 'obs_developer', 'pct_enabler': 'obs_enabler',
       'pct_adopter': 'obs_adopter', 'pct_beneficiary': 'obs_beneficiary',
       'has_developer': 'obs_developer', 'has_enabler': 'obs_enabler',
       'intense_adopter': 'obs_adopter'}
CLAIMS = ['uses_ai', 'theme_broad', 'theme_strict', 'theme_incidental',
          'chain_developer', 'chain_enabler', 'chain_adopter', 'chain_benefit_vague',
          'chain_unspecified', 'falsifiable_test']


def main():
    buf = io.StringIO()

    def w(s=''):
        buf.write(s + '\n')
        try:
            print(s)
        except UnicodeEncodeError:
            print(s.encode('ascii', 'replace').decode())

    H = pd.read_parquet(HOLD)
    H = H[H.value.notna() & (H.value > 0)].copy()
    SQ = pd.read_parquet(os.path.join(BENCH, 'stock_quarter.parquet'),
                         columns=['gvkey', 'q', 'bucket', 'mktcap'])
    BM = pd.read_parquet(os.path.join(BENCH, 'bucket_means.parquet'))

    H = H.merge(SQ, on=['gvkey', 'q'], how='left')
    H = H.merge(BM.rename(columns={m: 'exp_' + m for m in MEAS}),
                on=['q', 'bucket'], how='left')

    w('=' * 96)
    w('PORTFOLIO EXPOSURE PER FUND-QUARTER')
    w('=' * 96)
    w('  holding rows : %d, funds %d, fund-quarters %d'
      % (len(H), H.series_id.nunique(), H.groupby(['series_id', 'q']).ngroups))
    w('  rows with a benchmark bucket : %.1f%% of rows, %.1f%% of value'
      % (100 * H.bucket.notna().mean(),
         100 * H.loc[H.bucket.notna(), 'value'].sum() / H.value.sum()))
    w()

    # ---------------- fund-quarter exposure ----------------
    recs = []
    for (sid, q), g in H.groupby(['series_id', 'q'], sort=False):
        tot = g.value.sum()
        rec = {'series_id': sid, 'q': q, 'n_holdings': len(g),
               'port_value': tot,
               'report_date': g.report_date.iloc[0],
               'net_assets': g.net_assets.iloc[0],
               'series_name': g.series_name.iloc[0]}
        for m in MEAS:
            obs = g[OBS[m]].astype('boolean').fillna(False).astype(bool) & g[m].notna()
            cov = g.loc[obs, 'value'].sum() / tot if tot else np.nan
            rec[m + '_cov'] = cov
            if obs.any() and g.loc[obs, 'value'].sum() > 0:
                rec[m] = np.average(g.loc[obs, m].astype(float) * (100 if m.startswith(('has_', 'intense_')) else 1),
                                    weights=g.loc[obs, 'value'])
            else:
                rec[m] = np.nan
            # expected, on the same covered sleeve, benchmark rows only
            eb = obs & g['exp_' + m].notna()
            if eb.any() and g.loc[eb, 'value'].sum() > 0:
                sc = 100 if m.startswith(('has_', 'intense_')) else 1
                rec['exp_' + m] = np.average(
                    g.loc[eb, 'exp_' + m].astype(float) * sc, weights=g.loc[eb, 'value'])
                rec['act_on_bench_' + m] = np.average(
                    g.loc[eb, m].astype(float) * sc, weights=g.loc[eb, 'value'])
            else:
                rec['exp_' + m] = np.nan
                rec['act_on_bench_' + m] = np.nan
            rec['abn_' + m] = rec['act_on_bench_' + m] - rec['exp_' + m]
        recs.append(rec)
    E = pd.DataFrame(recs)

    w('  fund-quarters with exposure : %d' % len(E))
    w('  %-18s %10s %10s %10s' % ('measure', 'cov median', 'actual med', 'abnormal med'))
    for m in MEAS:
        w('  %-18s %9.1f%% %10.1f %10.1f'
          % (m.replace('pct_', ''), 100 * E[m + '_cov'].median(),
             E[m].median(), E['abn_' + m].median()))
    w()

    # ---------------- contemporaneous claim ----------------
    P = pd.read_parquet(PANEL)
    P['date'] = pd.to_datetime(P.date, errors='coerce')
    P = P[P.series_resolved & P.series_id.notna() & P.date.notna()]
    keep = ['series_id', 'date'] + [c for c in CLAIMS if c in P.columns]
    P = P[keep].sort_values('date')

    E['report_date'] = pd.to_datetime(E.report_date)
    E = E.sort_values('report_date')
    A = pd.merge_asof(E, P, left_on='report_date', right_on='date', by='series_id',
                      direction='backward')
    # a prospectus more than 16 months old is not the one in force (asof.py)
    A, n_expired = expire(A, 'report_date', [c for c in keep if c != 'series_id'])
    A['claim_date'] = A.date
    A['claim_age_days'] = (A.report_date - A.date).dt.days
    A['has_claim_asof'] = A.date.notna()

    w('-' * 96)
    w('CONTEMPORANEOUS CLAIM ALIGNMENT')
    w('-' * 96)
    w('  fund-quarters with a prospectus public by the report date : %d of %d (%.1f%%)'
      % (int(A.has_claim_asof.sum()), len(A), 100 * A.has_claim_asof.mean()))
    w('  ... matches older than %d days treated as unknown          : %d'
      % (MAX_AGE_DAYS, n_expired))
    w('  age of that prospectus at the report date: median %.0f days, p90 %.0f'
      % (A.claim_age_days.median(), A.claim_age_days.quantile(.9)))
    w()
    w('  claim prevalence in fund-quarters, contemporaneous vs ever:')
    ever = P.groupby('series_id')[[c for c in CLAIMS if c in P.columns]].any()
    w('  %-20s %14s %14s' % ('claim', 'fund-quarters', 'funds ever'))
    for c in CLAIMS:
        if c in A.columns:
            w('  %-20s %13d %14d' % (c, int(A[c].fillna(False).sum()),
                                     int(ever[c].sum()) if c in ever else -1))
    w()

    A.to_parquet(os.path.join(OUT, 'fund_quarter_exposure.parquet'), index=False)
    w('wrote out/fund_quarter_exposure.parquet (%d fund-quarters)' % len(A))

    with io.open(os.path.join(LOGS, 'c01_portfolio_exposure.txt'), 'w',
                 encoding='utf-8') as fh:
        fh.write(buf.getvalue())


if __name__ == '__main__':
    main()

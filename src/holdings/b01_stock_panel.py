"""The stock-quarter panel the characteristic-matched benchmark needs.

Protocol 4 of the project's own methods file: do not control linearly for size. Doing so
"flipped the talker coefficient from +5% raw to -16 pp", because AI-heavy stocks ARE the
mega-caps and the control absorbs exactly the variation the outcome is made of. Instead,
for each holding compute the AI measure of the average stock in the same
size-decile x sector bucket that quarter; a fund's expected exposure is the weighted
average of those, and its abnormal exposure is actual minus expected.

That is also what makes "beyond the usual suspects" well posed: a fund that simply owns
the biggest technology names gets an expected exposure as high as its actual one and
scores zero.

MARKET CAP IS US-ONLY, AND THAT IS A REAL LIMIT. Compustat Global has prices
(comp_g_secm.prccm) but no shares outstanding, and comp_g_funda has no csho, so a
size decile cannot be computed for foreign listings from what is on disk. The benchmark
therefore covers the US sleeve, and every result built on it must say so. The raw
portfolio shares - which need no benchmark - stay global.
"""
import io, os

import numpy as np
import pandas as pd

from src import paths as tp  # noqa: E402

OUT = tp.out('holdings')
LOGS = tp.logs('holdings')
MEAS = tp.out('holdings', 'firm_measures.parquet')
os.makedirs(OUT, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)
W = tp.data('wrds')

MEASURES = ['pct_developer', 'pct_enabler', 'pct_adopter', 'pct_beneficiary']
# the two-layer replacements: value share of the bucket in firms WITH the attribute
# the final measure set: sparse ones as a value share, dense one as a percentile
HAS = ['has_developer', 'has_enabler', 'intense_adopter']
ALLM = MEASURES + HAS


def gk(s):
    return pd.to_numeric(s, errors='coerce').astype('Int64').astype(str).str.zfill(6)


def main():
    buf = io.StringIO()

    def w(s=''):
        buf.write(s + '\n')
        try:
            print(s)
        except UnicodeEncodeError:
            print(s.encode('ascii', 'replace').decode())

    M = pd.read_parquet(MEAS)
    M['gvkey'] = gk(M.gvkey)

    # quarterly market cap, US: comp_secm has prccm and cshoq
    S = pd.read_parquet(os.path.join(W, 'comp_secm.parquet'),
                        columns=['gvkey', 'datadate', 'prccm', 'cshoq', 'curcdm'])
    S['gvkey'] = gk(S.gvkey)
    S['datadate'] = pd.to_datetime(S.datadate)
    S = S[S.datadate >= '2020-10-01']
    S['mktcap'] = S.prccm * S.cshoq
    S = S[S.mktcap.notna() & (S.mktcap > 0)]
    S['q'] = S.datadate.dt.to_period('Q').astype(str).str.lower()
    # one observation per gvkey-quarter: the last month in the quarter
    S = S.sort_values(['gvkey', 'q', 'datadate']).groupby(['gvkey', 'q']).tail(1)

    P = S[['gvkey', 'q', 'mktcap']].merge(
        M[['gvkey', 'conm', 'fic', 'sic4', 'sector', 'in_universe', 'is_enabler_core',
           'is_enabler_power'] + ALLM], on='gvkey', how='left')

    w('=' * 96)
    w('STOCK-QUARTER PANEL FOR THE CHARACTERISTIC BENCHMARK')
    w('=' * 96)
    w('  gvkey-quarters with a market cap : %d' % len(P))
    w('  distinct gvkeys                  : %d' % P.gvkey.nunique())
    w('  quarters                         : %s .. %s' % (P.q.min(), P.q.max()))
    w('  in the screened measure universe : %d (%.1f%%)'
      % (int(P.in_universe.fillna(False).sum()),
         100 * P.in_universe.fillna(False).mean()))
    w()

    # keep the screened universe for the benchmark buckets
    B = P[P.in_universe.fillna(False)].copy()
    B['size_decile'] = (B.groupby('q').mktcap
                         .transform(lambda s: pd.qcut(s.rank(method='first'), 10,
                                                      labels=False) + 1))
    B['bucket'] = B.sector.astype(str) + '_' + B.size_decile.astype(int).astype(str)

    w('-' * 96)
    w('BUCKETS: size decile x sector, formed each quarter')
    w('-' * 96)
    w('  stocks in the benchmark universe : %d gvkey-quarters, %d gvkeys'
      % (len(B), B.gvkey.nunique()))
    w('  buckets per quarter              : %.0f (median)'
      % B.groupby('q').bucket.nunique().median())
    cnt = B.groupby(['q', 'bucket']).size()
    w('  stocks per bucket                : median %.0f, p10 %.0f, min %d'
      % (cnt.median(), cnt.quantile(.1), cnt.min()))
    w()

    # value-weighted bucket means: what the AVERAGE stock of that kind looks like
    bm = (B.groupby(['q', 'bucket'])
           .apply(lambda g: pd.Series(
               {m: (np.average(g[m].astype(float).fillna(g[m].astype(float).mean()),
                               weights=g.mktcap)
                    if g[m].notna().any() else np.nan)
                for m in ALLM}), include_groups=False)
           .reset_index())
    w('  bucket-level expected exposure computed for %d (quarter, bucket) cells'
      % len(bm))
    w()
    w('  what a bucket mean looks like, 2025q4, largest-decile tech vs largest-decile other:')
    ex = bm[bm.q == '2025q4']
    for b in ('tech_10', 'other_10', 'finance_10', 'health_10'):
        r = ex[ex.bucket == b]
        if len(r):
            r = r.iloc[0]
            w('    %-12s developer %5.1f  enabler %5.1f  adopter %5.1f  beneficiary %5.1f'
              % (b, r.pct_developer, r.pct_enabler, r.pct_adopter, r.pct_beneficiary))
    w()
    w('  A fund that simply buys the biggest technology names inherits the tech_10')
    w('  figures as its EXPECTED exposure, so its abnormal exposure is near zero. That')
    w('  is the point of the benchmark, and it is what "beyond the usual suspects"')
    w('  means operationally.')

    B.to_parquet(os.path.join(OUT, 'stock_quarter.parquet'), index=False)
    bm.to_parquet(os.path.join(OUT, 'bucket_means.parquet'), index=False)
    w()
    w('wrote out/stock_quarter.parquet (%d rows)' % len(B))
    w('wrote out/bucket_means.parquet  (%d cells)' % len(bm))

    with io.open(os.path.join(LOGS, 'b01_stock_panel.txt'), 'w',
                 encoding='utf-8') as fh:
        fh.write(buf.getvalue())


if __name__ == '__main__':
    main()

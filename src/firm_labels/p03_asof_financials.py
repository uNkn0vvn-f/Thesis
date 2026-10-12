"""Revenue and market capitalisation as they stood at a cutoff, not as they stand now.

The company classifier's description of each company carries `annual revenue` and
`market capitalisation` (src/firm_labels/c01_classify.py, `describe`). For a label that has
to be fixed before a cutoff, current figures would leak: for companies with no filing text
and no patents, name, industry, revenue and market capitalisation are all the model sees,
and a 2025 market capitalisation largely records which companies the AI boom rewarded.

WHAT IT DOES. Builds both figures from the last fiscal year whose results were PUBLIC at
the cutoff. An annual figure is not public on the fiscal year end - the 10-K follows. The
lag is set by `PUB_LAG_DAYS` and defaults to 90, which is the statutory maximum for a
non-accelerated filer and therefore conservative: it can only ever make the evidence
older, never newer than it really was.

    CUTOFF          yyyy-mm-dd, the shock date
    FIN_OUT         directory for financials_asof.parquet
    PUB_LAG_DAYS    days after fiscal year end before the figures count as public (90)
"""
import io, os

import pandas as pd

from src import paths as tp  # noqa: E402

W = tp.data('wrds')
UNI = os.environ.get('UNIVERSE') or tp.out('firm_labels', 'firm_universe.parquet')
CUTOFF = pd.Timestamp(os.environ['CUTOFF'])
OUTDIR = os.environ['FIN_OUT']
LAG = int(os.environ.get('PUB_LAG_DAYS', '90'))
os.makedirs(OUTDIR, exist_ok=True)


def gk(s):
    """Compustat gvkeys as six-digit strings."""
    return s.astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(6)


def main():
    """Write each universe firm's last revenue and market capitalisation public at CUTOFF."""
    lines = []

    def w(s=''):
        lines.append(s)
        print(s)

    U = pd.read_parquet(UNI, columns=['gvkey', 'name', 'value_all', 'revenue', 'mktcap'])
    U['gvkey'] = gk(U.gvkey)
    U = U.rename(columns={'revenue': 'revenue_now', 'mktcap': 'mktcap_now'})

    # The last fiscal year whose figures were public at the cutoff: period end plus the
    # publication lag.
    fa = pd.read_parquet(os.path.join(W, 'comp_funda.parquet'),
                         columns=['gvkey', 'datadate', 'fyear', 'revt', 'sale',
                                  'csho', 'prcc_f'])
    fa['gvkey'] = gk(fa.gvkey)
    fa['datadate'] = pd.to_datetime(fa.datadate)
    fa['public_on'] = fa.datadate + pd.Timedelta(days=LAG)
    pre = fa[fa.public_on <= CUTOFF].sort_values(['gvkey', 'datadate'])
    last = pre.groupby('gvkey').tail(1).copy()
    last['revenue'] = last.revt.fillna(last.sale)
    last['mktcap'] = last.csho * last.prcc_f
    A = last[['gvkey', 'fyear', 'datadate', 'revenue', 'mktcap']].rename(
        columns={'fyear': 'fin_fyear', 'datadate': 'fin_datadate'})

    M = U.merge(A, on='gvkey', how='left')
    M.to_parquet(os.path.join(OUTDIR, 'financials_asof.parquet'), index=False)

    w('=' * 88)
    w('AS-OF FINANCIALS  cutoff %s, publication lag %d days' % (CUTOFF.date(), LAG))
    w('=' * 88)
    w('  universe firms                     : %d' % len(M))
    w('  with as-of revenue                 : %d (%.1f%%)'
      % (M.revenue.notna().sum(), 100 * M.revenue.notna().mean()))
    w('  with as-of market cap              : %d (%.1f%%)'
      % (M.mktcap.notna().sum(), 100 * M.mktcap.notna().mean()))
    w('  latest fiscal year used            : %s' % (M.fin_fyear.max()))
    w('  latest datadate used               : %s   (public by %s)'
      % (M.fin_datadate.max().date(),
         (M.fin_datadate.max() + pd.Timedelta(days=LAG)).date()))
    assert (M.fin_datadate.dropna() + pd.Timedelta(days=LAG) <= CUTOFF).all()
    w()
    # Report how far the as-of market capitalisation is from the current one.
    b = M[M.revenue.notna() & M.mktcap_now.notna() & (M.mktcap_now > 0)].copy()
    b['ratio'] = b.mktcap / b.mktcap_now
    w('  HOW MUCH THE LEAK WAS WORTH, market capitalisation as-of vs as-built:')
    w('    median as-of / current ratio     : %.2f' % b.ratio.median())
    w('    firms whose market cap more than doubled since : %d (%.1f%% of them)'
      % (int((b.ratio < .5).sum()), 100 * (b.ratio < .5).mean()))
    w()
    w('  the ten firms by held value whose market cap moved most:')
    t = b.sort_values('value_all', ascending=False).head(400)
    t = t.reindex(t.ratio.sub(1).abs().sort_values(ascending=False).index).head(10)
    for _, r in t.iterrows():
        w('    %-32s as-of $%8.0fm -> now $%9.0fm  (x%.1f)'
          % (str(r['name'])[:32], r.mktcap, r.mktcap_now, 1 / r.ratio if r.ratio else 0))
    with io.open(os.path.join(OUTDIR, 'financials_asof.txt'), 'w',
                 encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()

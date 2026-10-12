"""An AI factor built from what firms ARE, not from the codes that define the claim.

The performance test needs to price the AI exposure a fund bought before asking whether
the fund added anything. A factor built from the SIC list that
DEFINES the enabler claim would make the control and the treatment two readings of the
same industry map - close to regressing a variable on itself.

This builds the factor from the direct firm label instead:

    AIREV   value-weighted return of firms whose estimated AI REVENUE SHARE is at least
            50%, minus the value-weighted market
    AITOP   the same for the top decile of AI revenue share among firms with any

That is still a control constructed from holdings characteristics, and it must be, because
the question is whether a fund beat the AI stocks it owns. What it is NOT is a control
constructed from the same text taxonomy that assigns the claim: the label comes from
company annual reports and patent portfolios, the claim comes from fund prospectuses, and
nothing joins them except the fund's own portfolio.

Both are reported beside the SIC-based `enab` factor so the difference is visible rather
than asserted.

THE PORTFOLIO IS REFORMED EVERY YEAR. Membership held fixed from one recent label would
put a firm that became an AI company in 2024 in the 2020 portfolio, and the factor's return
would be that of a portfolio picked with knowledge of its own future. Membership is
therefore taken from the annual company-label vintages (`python run.py prepare`): one label
per calendar year, each built only from the filings, patents and financial statements
public on 31 December of the year before. The decile cut is recomputed inside each vintage,
so it too is a number an investor had.

WHAT THE VINTAGES DO NOT FIX. The set of firms
eligible for a label at all is the holdings universe - firms held by a claiming fund at
some point in 2021-2025 - and that set is not point-in-time. A firm is in the 2020
portfolio partly because funds bought it later. Labelling all of CRSP instead would cost
several times the API budget for a control variable, so the limitation is reported rather
than removed; it is second order here because the factor is a control, and because the
selection runs through fund holdings rather than through firm returns. Second, a firm with
no 10-K in its vintage window is classified from its name, industry code and patents by a
model trained through 2025, which is a residual channel for the future. The share of each
year's portfolio that had a filing is printed in the log.

    python -m src.analysis.flows.ai_factor     writes analysis/alpha/out/factors.parquet
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from src import paths as tp  # noqa: E402
import io

import numpy as np
import pandas as pd

OUT = tp.out('analysis', 'alpha', 'out')
LOGS = tp.out('analysis', 'alpha', 'logs')

W = tp.data('wrds')
VINT = tp.out('firm_labels', 'vintages', '02_label')
EVID = tp.out('firm_labels', 'vintages', '00_evidence')
FAC = tp.out('analysis', 'factors', 'out', 'factors_monthly.parquet')

THRESH = 50.0
DEST = 'factors.parquet'


def load_labels():
    """One label per calendar year.

    Returned long, with `yr` the calendar year the label governs, so the merge on to the
    return panel is on (gvkey, year) and a firm can enter or leave the portfolio."""
    rows, pending = [], []
    for y in sorted(int(d) for d in os.listdir(VINT) if d.isdigit()):
        lab = os.path.join(VINT, str(y), 'firm_ai_labels.parquet')
        if not os.path.exists(lab):
            # the driver creates the directory before the label lands, so a half-built
            # vintage is skipped - but named, because a year silently missing here would
            # leave those months with no portfolio at all and nothing would say so
            pending.append(y)
            continue
        x = pd.read_parquet(lab, columns=['gvkey', 'name', 'ai_rev'])
        x['gvkey'] = x.gvkey.astype(str)
        x['yr'] = y
        # Not every firm filed an annual report in the twelve months before the cutoff, and
        # one that did not is classified from its name, its industry code and its patents.
        # The classifier is a 2025 model, so for those firms its own training is a residual
        # channel through which the future can enter. `has_10k` marks the firms with a
        # filing (10-K, 20-F or 40-F) in the vintage's accession list.
        acc = os.path.join(EVID, str(y), 'accessions.parquet')
        seen = set()
        if os.path.exists(acc):
            seen = set(pd.read_parquet(acc, columns=['gvkey']).gvkey.astype(str))
        x['has_10k'] = x.gvkey.isin(seen)
        rows.append(x)
    assert rows, 'no vintages in %s - build them first (python run.py prepare)' % VINT
    if pending:
        print('  WARNING: no label built yet for %s; those months get no AI portfolio'
              % ', '.join(str(y) for y in pending))
    L = pd.concat(rows, ignore_index=True)
    assert not L.duplicated(['gvkey', 'yr']).any(), 'a gvkey appears twice in one vintage'
    return L, sorted(L.yr.unique())


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    buf = io.StringIO()

    def w(s=''):
        buf.write(s + '\n')
        try:
            print(s)
        except UnicodeEncodeError:
            print(s.encode('ascii', 'replace').decode())

    L, years = load_labels()
    CL = pd.read_parquet(os.path.join(W, 'ccm_linktable.parquet'))
    CL['gvkey'] = CL.gvkey.astype(str)
    CL = CL[CL.linktype.isin(['LU', 'LC']) & CL.linkprim.isin(['P', 'C'])]
    CL['linkdt'] = pd.to_datetime(CL.linkdt)
    CL['linkenddt'] = pd.to_datetime(CL.linkenddt).fillna(pd.Timestamp('2100-01-01'))

    S = pd.read_parquet(os.path.join(W, 'crsp_msf_v2.parquet'),
                        columns=['permno', 'mthcaldt', 'mthret', 'mthcap',
                                 'sharetype', 'securitytype', 'securitysubtype',
                                 'issuertype', 'primaryexch'])
    S['mthcaldt'] = pd.to_datetime(S.mthcaldt)
    S['month'] = pd.to_datetime(S.mthcaldt).dt.to_period('M').dt.to_timestamp()
    S = S[(S.sharetype == 'NS') & (S.securitytype == 'EQTY')
          & (S.securitysubtype == 'COM') & S.issuertype.isin(['CORP', 'ACOR'])
          & S.primaryexch.isin(['N', 'A', 'Q'])]
    S['mthret'] = pd.to_numeric(S.mthret, errors='coerce')
    S = S.sort_values(['permno', 'month'])
    if S.duplicated(['permno', 'month']).any():
        raise ValueError('Duplicate CRSP stock-months cannot define a unique lagged market cap')
    month_number = S.month.dt.year * 12 + S.month.dt.month
    gap = month_number - month_number.groupby(S.permno).shift()
    # A missing return does not remove the previous month's observed market cap.
    # Missing calendar months must not turn an older observation into a one-month lag.
    S['me_lag'] = S.groupby('permno').mthcap.shift().where(gap.eq(1))
    S = S[S.mthret.notna() & S.mthcap.notna() & (S.mthcap > 0)
          & S.me_lag.notna() & (S.me_lag > 0)]

    J = S.merge(CL[['gvkey', 'permno', 'linkdt', 'linkenddt']], on='permno', how='left')
    J = J[(J.mthcaldt >= J.linkdt) & (J.mthcaldt <= J.linkenddt)]
    J['yr'] = J.month.dt.year
    J = J.merge(L, on=['gvkey', 'yr'], how='left')

    w('=' * 96)
    w('AN AI FACTOR FROM THE FIRM LABEL  [ANNUAL VINTAGES]')
    w('=' * 96)
    w('  CRSP stock-months %d; matched to a gvkey %.1f%%; carrying an AI label %.1f%%'
      % (len(S), 100 * J.gvkey.notna().mean(), 100 * J.ai_rev.notna().mean()))

    # The decile cut belongs to the vintage: an investor forming the 2021 portfolio
    # knew the 2021 distribution of AI revenue, not the 2025 one.
    #
    # It is taken on the FORMATION date. The investable set of
    # year y is the firms with a CRSP return and a lagged market cap in January of y,
    # one record per firm; the cut is the 90th percentile of their AI sales share, and a
    # member holds its place through December (leaving only if its returns stop). A cut
    # taken over every stock-month of the year would count a firm once per month it
    # traded and let firms listing later in the year move January's cut; a firm that
    # lists in June waits for the next January.
    # `airev` (at least half of revenue from AI) is formed the same way.
    F0 = (J[(J.month.dt.month == 1) & J.ai_rev.notna()]
          .drop_duplicates(['gvkey', 'yr'])[['gvkey', 'yr', 'ai_rev']])
    cut = F0[F0.ai_rev > 0].groupby('yr').ai_rev.quantile(.9).rename('thr_top')
    F0 = F0.merge(cut, on='yr', how='left')
    F0['f_top'] = F0.ai_rev >= F0.thr_top
    F0['f_rev'] = F0.ai_rev >= THRESH
    J = J.merge(F0[['gvkey', 'yr', 'f_top', 'f_rev']], on=['gvkey', 'yr'], how='left')
    J['in_aitop'] = J.f_top.astype('boolean').fillna(False).astype(bool)
    J['in_airev'] = J.f_rev.astype('boolean').fillna(False).astype(bool)
    w('  formed each January on that January\'s firms, one record per firm')
    # AITOPEV is the strict version: the same decile, restricted to firms whose label
    # rests on a 10-K filed before the cutoff. It drops two kinds of member - foreign
    # issuers, which file 20-F and never a 10-K (TSMC, Samsung, Baidu), and firms that
    # listed during the year itself (Palantir, UiPath, AppLovin in 2021) - whose label
    # could only have come from the classifier reading a name. Every member of AITOPEV
    # was classified from evidence an investor could have read; the cost is a smaller
    # portfolio, and a worse control for funds that held exactly those firms, so both
    # are carried and priced.
    J['in_aitop_ev'] = J.in_aitop & J.has_10k
    w('  the portfolio is reformed each January from the label public the December')
    w('  before; a firm enters when its own reading crosses the cut, not when the')
    w('  2025 reading does.')
    w()
    w('  %-6s %9s %10s %9s %9s %10s %11s'
      % ('year', 'labelled', 'decile cut', 'in airev', 'in aitop', 'new to top',
         'top w/ 10-K'))
    prev = set()
    for y in years:
        g = J[J.yr == y]
        top = set(g.loc[g.in_aitop, 'gvkey'])
        ev = g.loc[g.in_aitop & g.has_10k, 'gvkey'].nunique()
        w('  %-6d %9d %9.1f%% %9d %9d %10d %10.0f%%'
          % (y, g.gvkey[g.ai_rev.notna()].nunique(),
             cut.get(y, float('nan')),
             g.loc[g.in_airev, 'gvkey'].nunique(),
             len(top), len(top - prev) if prev else 0,
             100.0 * ev / len(top) if top else float('nan')))
        prev = top
    w()

    def vw(d):
        return 100 * np.average(d.mthret, weights=d.me_lag)

    mkt = S.groupby('month').apply(vw, include_groups=False).rename('r_mkt')
    out = {'r_mkt': mkt}
    ports = [('airev', 'in_airev'), ('aitop', 'in_aitop'), ('aitopev', 'in_aitop_ev')]
    for nm, col in ports:
        sub = J[J[col]]
        out['r_' + nm] = sub.groupby('month').apply(vw, include_groups=False)
        out['n_' + nm] = sub.groupby('month').permno.nunique()
    P = pd.DataFrame(out).reset_index()
    for nm, _ in ports:
        P[nm] = P['r_' + nm] - P.r_mkt

    F = pd.read_parquet(FAC)
    F['month'] = pd.to_datetime(F.month)
    keep = ['month'] + [nm for nm, _ in ports] + ['n_' + nm for nm, _ in ports]
    M = F.merge(P[keep], on='month',
                how='left')

    w('-' * 96)
    w('THE FACTORS')
    w('-' * 96)
    w('  %-8s %9s %9s %9s %9s %9s'
      % ('factor', 'mean', 'sd', 'min', 'max', 'months'))
    names = ['mktrf', 'smb', 'hml', 'umd', 'enab', 'tech'] + [n for n, _ in ports]
    for c in names:
        v = M[c].dropna()
        w('  %-8s %9.3f %9.3f %9.3f %9.3f %9d'
          % (c, v.mean(), v.std(), v.min(), v.max(), len(v)))
    w()
    w('  stocks per month: ' + ', '.join(
      '%s median %d' % (n, M['n_' + n].median()) for n, _ in ports))
    w()
    C = M[['mktrf', 'smb', 'hml', 'umd', 'enab'] + [n for n, _ in ports]].corr()
    w('  correlations:')
    w('  %-8s' % '' + ''.join('%9s' % c for c in C.columns))
    for r in C.index:
        w('  %-8s' % r + ''.join('%9.2f' % C.loc[r, c] for c in C.columns))
    w()
    w('  airev correlates %.2f with the SIC-based enab factor. They are close but not'
      % C.loc['airev', 'enab'])
    w('  the same thing, and only airev is built independently of the industry map that')
    w('  defines the enabler claim.')
    w()
    w('  excess-of-market return by year, % compounded within year:')
    M['yr'] = M.month.dt.year
    cols = ['enab'] + [n for n, _ in ports]
    w('  %-6s %10s' % ('year', 'market') + ''.join('%10s' % c for c in cols))
    for y, g in M.groupby('yr'):
        def cm(c):
            v = g[c].dropna()
            return np.nan if len(v) < len(g) else 100 * ((1 + v / 100).prod() - 1)
        f = lambda x: '%10s' % 'n/a' if not np.isfinite(x) else '%10.1f' % x
        w('  %-6d %10.1f' % (y, cm('mktrf')) + ''.join(f(cm(c)) for c in cols))
    w()

    M.drop(columns='yr').to_parquet(os.path.join(OUT, DEST), index=False)
    w('wrote out/%s' % DEST)
    with io.open(os.path.join(LOGS, 'ai_factor.txt'), 'w', encoding='utf-8') as fh:
        fh.write(buf.getvalue())


if __name__ == '__main__':
    main()

"""Compare thematic claimants with a nonclaimant AI-revenue benchmark.

The portfolio score aggregates estimated company AI revenue shares. The comparison group
sets the benchmark rather than a rank within thematic claimants:

    BELOW = claims an AI theme, and holds no more AI revenue than the 75th percentile
            of funds that claim nothing at all
    ABOVE = claims an AI theme, and holds more

Neither group establishes whether a fund's statements are misleading.

The full-sample bar is used only for the descriptive fund-level table. Monthly flow and
return comparisons use a separate bar at each month: the 75th percentile of fund-level
means over known nonclaimant quarters whose holdings were filed strictly before that
month. Adding a later filing must not change an earlier month's classification.

The fund's score is its latest eligible holdings record filed before the month, not an
expanding mean; the report must be at most 366 days old. The claim is the prospectus in
force that month. Missing holdings or an unavailable historical bar leave the comparison
unclassified. This is a disclosure-portfolio screen, not proof of a misleading claim.

Each flow block reports an approximate minimum detectable difference at 80% power;
this is not a confidence bound or a guarantee that an effect would be detected. The
alphas are those of equal-weighted monthly portfolios of each group (overlays excluded,
net of fees), on Carhart's four factors and on Carhart plus the AI factor `aitop`
(src.analysis.flows.ai_factor), with Newey-West standard errors.
"""
import io
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from src import paths as tp  # noqa: E402

import numpy as np
import pandas as pd

from src.analysis.flows.estimation import (ols_cluster, MIN_TNA, OVERLAY, absorb,  # noqa: E402
                                           make_dims, absorbed_k, nw)
from src.analysis.flows.claim_rungs import calendar_flow_lag                      # noqa: E402

OUT = tp.out('analysis', 'alignment', 'out')
LOGS = tp.out('analysis', 'alignment', 'logs')
EXP = tp.out('exposure', 'fund_quarter_exposure.parquet')
LAD = tp.out('analysis', 'flow_ladder', 'out', 'fund_month_ladder.parquet')
RET = tp.out('analysis', 'monthly_returns', 'out', 'fund_month_returns.parquet')
FAC = tp.out('analysis', 'alpha', 'out', 'factors.parquet')

C4 = ['mktrf', 'smb', 'hml', 'umd']
MIN_LOOK = 50
FRESH_DAYS = 366        # holdings dated more than a year before the month do not judge it


def historical_bars(exposure, months):
    """Monthly p75 of available nonclaimant fund means, excluding same-day filings."""
    eligible = (~(exposure.uses_ai | exposure.theme_broad)
                & exposure.claim_age_days.notna())
    controls = exposure.loc[eligible, ['series_id', 'filing_date', 'ai_rev']].copy()
    controls['filing_date'] = pd.to_datetime(controls.filing_date)
    controls = controls.dropna(subset=['series_id', 'filing_date', 'ai_rev'])
    controls = controls[np.isfinite(controls.ai_rev)].sort_values(
        ['filing_date', 'series_id', 'ai_rev'], kind='stable')
    groups = controls.groupby('series_id', sort=False)
    controls['fund_mean'] = groups.ai_rev.cumsum() / (groups.cumcount() + 1)
    dates = controls.filing_date.to_numpy()
    values = {}
    for month in sorted(pd.DatetimeIndex(months).dropna().unique()):
        stop = dates.searchsorted(month.to_datetime64(), side='left')
        available = controls.iloc[:stop].drop_duplicates('series_id', keep='last')
        values[month] = available.fund_mean.quantile(.75) if len(available) else np.nan
    return pd.Series(values, name='bar', dtype=float).rename_axis('month')


def holdings_history(exposure):
    """Resolve same-day reports before an as-of join, independent of input order."""
    history = exposure[['series_id', 'report_date', 'filing_date', 'ai_rev']].copy()
    history['report_date'] = pd.to_datetime(history.report_date)
    history['filing_date'] = pd.to_datetime(history.filing_date)
    history = history.dropna(subset=['filing_date'])
    keys = ['series_id', 'filing_date', 'report_date']
    conflicts = history.groupby(keys, dropna=False).ai_rev.nunique(dropna=False) > 1
    if conflicts.any():
        raise ValueError('Conflicting AI exposures for the same fund, filing and report date')
    # Several quarters may be filed together; the newest report describes that date.
    return (history.sort_values(['filing_date', 'series_id', 'report_date'],
                                kind='stable', na_position='first')
            .drop_duplicates(['series_id', 'filing_date'], keep='last'))


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

    E = pd.read_parquet(EXP)
    E = E[E.labelled >= MIN_LOOK].sort_values(['series_id', 'q']).copy()
    E['any_ai'] = E.uses_ai | E.theme_broad
    # the bar is set on fund-quarters KNOWN to make no claim: an expired or missing
    # prospectus is unknown, not 'no claim' (claim_age_days is empty exactly then)
    ctrl = E[~E.any_ai & E.claim_age_days.notna()].groupby('series_id').ai_rev.mean()
    bar = float(ctrl.quantile(.75))

    w('=' * 96)
    w('AI THEME CLAIMANTS AGAINST THE AI REVENUE HELD BY FUNDS THAT CLAIM NOTHING')
    w('=' * 96)
    w('  funds claiming nothing hold, value-weighted, an AI revenue share of')
    w('  median %.1f%%, p75 %.1f%%, p90 %.1f%% (%d funds)'
      % (ctrl.median(), ctrl.quantile(.75), ctrl.quantile(.9), len(ctrl)))
    w('  DESCRIPTIVE FULL-SAMPLE BAR : %.1f%% (not used for historical classification)' % bar)
    w()

    F = (E.groupby('series_id')
         .agg(ai_all=('ai_rev', 'mean'), theme=('theme_broad', 'any'),
              uses=('uses_ai', 'any'), nq=('q', 'nunique'),
              nm=('series_name', 'last')).reset_index())
    import warnings
    warnings.filterwarnings('ignore', message='This pattern is interpreted')
    F['is_overlay'] = F.nm.fillna('').str.contains(OVERLAY, regex=True)
    # a theme claimant is judged on the quarters in which it claims
    F['ai_rev'] = F.series_id.map(E[E.theme_broad].groupby('series_id').ai_rev.mean())
    F['ai_rev'] = F.ai_rev.fillna(F.ai_all)
    F['below'] = F.theme & (F.ai_rev <= bar)
    w('  %-26s %8s %14s %12s %10s'
      % ('group', 'funds', 'median AI rev', 'below bar', '% below'))
    for lab, m in (('claims an AI theme', F.theme),
                   ('uses AI only', F.uses & ~F.theme),
                   ('no AI claim', ~F.theme & ~F.uses)):
        s = F[m]
        w('  %-26s %8d %13.1f%% %12d %9.0f%%'
          % (lab, len(s), s.ai_rev.median(), int((s.ai_rev <= bar).sum()),
             100 * (s.ai_rev <= bar).mean()))
    w()
    for lab, s in (('all theme claimants', F[F.theme]),
                   ('overlays excluded', F[F.theme & ~F.is_overlay])):
        w('  %-24s %d of %d (%.0f%%) hold no more AI revenue than the'
          % (lab, int(s.below.sum()), len(s), 100 * s.below.mean()))
        w('  %-24s 75th-percentile benchmark for funds that claim nothing.' % '')
    w()

    # ---------------- point-in-time, on the CURRENT claim ----------------
    # A fund-month is treated only while the prospectus in force claims an AI theme (rung 3,
    # 4 or 5 that month), and it is judged on the latest fund-quarter FILED before the month,
    # dated within FRESH_DAYS, with half its value labelled. The control is only fund-months
    # whose prospectus in force makes no AI claim (rung 0). Process claims (rung 2) and AI
    # names without a claim (rung 1) are groups of their own, and a theme claim with no such
    # holdings or historical benchmark is "comparison unavailable", not a control.
    H = holdings_history(E)
    L = pd.read_parquet(LAD, columns=['series_id', 'month', 'rung', 'rung_known',
                                      'na_beg_recon', 'nf_beg', 'ret', 'sponsor',
                                      'series_name']).sort_values('month')
    L['month'] = pd.to_datetime(L.month)
    A = pd.merge_asof(L, H, left_on='month', right_on='filing_date', by='series_id',
                      direction='backward', allow_exact_matches=False)
    monthly_bars = historical_bars(E, A.month)
    A['bar'] = A.month.map(monthly_bars)
    age = (A.month - A.report_date).dt.days
    fresh = A.report_date.notna() & age.between(0, FRESH_DAYS) \
        & A.ai_rev.notna()
    comparable = fresh & A.bar.notna()
    theme_now = A.rung_known & A.rung.isin([3, 4, 5])
    A['group'] = np.select(
        [~A.rung_known, A.rung == 0, A.rung == 1, A.rung == 2,
         theme_now & comparable & (A.ai_rev <= A.bar),
         theme_now & comparable & (A.ai_rev > A.bar),
         theme_now],
        ['unknown', 'control', 'name only', 'process', 'below', 'above',
         'theme, comparison unavailable'], default='unknown')
    A['claimant'] = A.group.isin(['below', 'above'])
    A['below'] = np.where(A.claimant, A.group == 'below', np.nan)
    A['pit'] = np.where(A.claimant, A.ai_rev, np.nan)
    A['is_overlay'] = A.series_name.fillna('').str.contains(OVERLAY, regex=True)
    # every treated month currently claims a theme, and every control month claims nothing
    assert not (A.claimant & ~theme_now).any()
    assert not ((A.group == 'control') & (A.rung != 0)).any()
    born = A.groupby('series_id').month.min()
    A['age_m'] = ((A.month.dt.year - A.series_id.map(born).dt.year) * 12
                  + (A.month.dt.month - A.series_id.map(born).dt.month))

    K = A[A.claimant]
    w('  point-in-time: below %d funds (median %.1f%%), above %d (median %.1f%%)'
      % (K[K.below == True].series_id.nunique(),      # noqa: E712
         K.loc[K.below == True, 'pit'].median(),      # noqa: E712
         K[K.below == False].series_id.nunique(),     # noqa: E712
         K.loc[K.below == False, 'pit'].median()))    # noqa: E712
    w('  fund-months by group (prospectus in force that month):')
    for gname in ('control', 'below', 'above', 'theme, comparison unavailable', 'process',
                  'name only', 'unknown'):
        s = A[A.group == gname]
        w('    %-28s %8d fund-months %6d funds' % (gname, len(s), s.series_id.nunique()))
    w()

    # ---------------- flows ----------------
    D = A[A.rung_known & A.sponsor.notna()].copy()
    D['nf'] = D.nf_beg.replace([np.inf, -np.inf], np.nan)
    D = D[D.na_beg_recon.notna() & (D.na_beg_recon >= MIN_TNA) & D.nf.notna()
          & D.ret.notna()].sort_values(['series_id', 'month'])
    D['nf_lag'] = calendar_flow_lag(D)
    D['lsize'] = np.log(D.na_beg_recon)
    D['lage'] = np.log1p(D.age_m)
    D = D[D.nf_lag.notna()]
    for c in ('nf', 'nf_lag'):
        lo = D.groupby('month')[c].transform(lambda s: s.quantile(.01))
        hi = D.groupby('month')[c].transform(lambda s: s.quantile(.99))
        D[c + '_w'] = D[c].clip(lo, hi)
    D['w_below'] = (D.group == 'below').astype(float)
    D['w_above'] = (D.group == 'above').astype(float)
    # the other claim groups get their own terms, so the omitted group is rung 0 alone
    D['w_unobs'] = (D.group == 'theme, comparison unavailable').astype(float)
    D['w_process'] = (D.group == 'process').astype(float)
    D['w_name'] = (D.group == 'name only').astype(float)
    XN = ['w_below', 'w_above', 'w_unobs', 'w_process', 'w_name', 'lsize', 'ret',
          'nf_lag_w', 'lage']

    w('-' * 96)
    w('1. FLOWS  (pp/month; omitted = fund-months whose prospectus in force claims nothing)')
    w('-' * 96)
    rows = []
    for nm, sub in (('all claimants', D), ('overlays excluded', D[~D.is_overlay]),
                    ('age >= 12 months', D[D.age_m >= 12])):
        s2, dims = make_dims(sub, [['sponsor', 'month']])
        Y = absorb(s2[['nf_w']].values, dims)[:, 0]
        X = absorb(s2[XN].values, dims)
        b, se, V, G = ols_cluster(Y, X, s2.series_id.values,
                                  n_absorbed=absorbed_k(dims, s2.series_id.values))
        d = b[0] - b[1]
        Lv = np.zeros(len(XN)); Lv[0] = 1; Lv[1] = -1
        sd = float(np.sqrt(Lv @ V @ Lv))
        w('  %-22s below %7.3f (%5.2f)   above %7.3f (%5.2f)   diff %7.3f (%5.2f)'
          % (nm, b[0], b[0] / se[0], b[1], b[1] / se[1], d, d / sd))
        # 2.80 = 1.96 + 0.84: the two-sided 5% critical value PLUS the 80th percentile
        # of the standard normal. 1.96 alone is the significance threshold, not a power
        # calculation, and using it understates the detectable effect by 43%.
        w('  %-22s approximate MDE at 80%% power: %.2fpp/month'
          % ('', 2.80 * sd))
        rows.append({'test': 'flows', 'cut': nm, 'below': b[0], 'se_below': se[0],
                     'above': b[1], 'se_above': se[1], 'diff': d, 'se_diff': sd,
                     'n': len(s2), 'n_funds': G,
                     'month_start': s2.month.min(), 'month_end': s2.month.max()})
    w()

    # ---------------- alpha ----------------
    R = pd.read_parquet(RET)
    R['month'] = pd.to_datetime(R.month)
    FA = pd.read_parquet(FAC)
    FA['month'] = pd.to_datetime(FA.month)
    # both models on the same months, those where the AI factor exists, so the change from
    # Carhart to Carhart + aitop is the factor's and not the sample's
    FA = FA.dropna(subset=C4 + ['aitop', 'rf']).copy()
    P = R.merge(A[['series_id', 'month', 'group', 'is_overlay']],
                on=['series_id', 'month'], how='inner')
    P = P[P.ret.notna() & ~P.is_overlay]
    key = np.where(P.group == 'control', 'control',
                   np.where(P.group == 'below', 'below',
                            np.where(P.group == 'above', 'above', 'other')))
    port = []
    for (m, k), g in P.assign(key=key).groupby(['month', 'key']):
        if len(g) < 5:
            continue
        port.append({'month': m, 'key': k, 'p': g.ret.mean()})
    PP = pd.DataFrame(port).merge(FA, on='month', how='left')
    PP['ex'] = PP.p - PP.rf

    w('-' * 96)
    w('2. ALPHA  (overlays excluded, equal-weighted, net of fees, NW 3)')
    w('-' * 96)
    for mname, xs in (('Carhart 4', C4), ('Carhart 4 + aitop', C4 + ['aitop'])):
        w('  %s' % mname)
        got = {}
        for k in ('below', 'above', 'control'):
            g = PP[PP.key == k].dropna(subset=['ex'] + xs).sort_values('month')
            if len(g) < 18:
                continue
            b, se = nw(g.ex.values, g[xs].values)
            got[k] = g.set_index('month').ex
            w('  %-22s %8.3f %8.3f %7.2f  %6d months'
              % (k, b[0], se[0], b[0] / se[0] if se[0] else np.nan, len(g)))
            rows.append({'test': 'alpha', 'cut': mname, 'group': k, 'alpha': b[0],
                         'se': se[0], 'n_months': len(g),
                         'month_start': g.month.min(), 'month_end': g.month.max()})
        if 'below' in got and 'above' in got:
            s = ((got['below'] - got['above']).dropna().rename('ex').to_frame()
                 .join(FA.set_index('month')).dropna(subset=['ex'] + xs))
            b, se = nw(s.ex.values, s[xs].values)
            w('  %-22s %8.3f %8.3f %7.2f  %6d months'
              % ('below - above', b[0], se[0],
                 b[0] / se[0] if se[0] else np.nan, len(s)))
            rows.append({'test': 'alpha', 'cut': mname, 'group': 'below - above',
                         'alpha': b[0], 'se': se[0], 'n_months': len(s),
                         'month_start': s.index.min(), 'month_end': s.index.max()})
        w()

    T = pd.DataFrame(rows)
    T.to_parquet(os.path.join(OUT, 'alignment.parquet'), index=False)
    tables = tp.RESULTS / 'tables'
    tables.mkdir(parents=True, exist_ok=True)
    T.to_csv(tables / 'alignment_flows.csv', index=False,
             float_format='%.12g', lineterminator='\n')
    F.to_parquet(os.path.join(OUT, 'alignment_fund.parquet'), index=False)
    pd.DataFrame([{'bar': bar}]).to_parquet(os.path.join(OUT, 'bar.parquet'),
                                            index=False)
    monthly_bars.reset_index().to_parquet(os.path.join(OUT, 'bar_by_month.parquet'), index=False)
    w('wrote out/alignment.parquet, out/alignment_fund.parquet, out/bar.parquet, out/bar_by_month.parquet')
    with io.open(os.path.join(LOGS, 'alignment.txt'), 'w', encoding='utf-8') as fh:
        fh.write(buf.getvalue())


if __name__ == '__main__':
    main()

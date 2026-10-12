"""Monthly flows by claim type and holdings alignment.

Rung 0 alone is the omitted category. Name-only funds (rung 1) and named but
unconfirmed theme claims (rung 4) each have their own coefficient and descriptive path.

    net flow rate (% of beginning-of-month TNA, winsorised 1/99 within month)
      ~ rung dummies + log size + own return + lagged flow + log fund age
      absorbing SPONSOR x MONTH,  standard errors clustered by fund

THE FLOW VARIABLE IS CHOSEN BY THE ACCOUNTING IDENTITY, not asserted. N-PORT's
REDEMPTION_FLOW is signed negative by 18.2% of filers and positive by 62.2%, so
`sales - redemption` ADDS redemptions for a fifth of filings. Rolling the PREVIOUS
quarter-end net assets forward through three months and comparing with the observed
quarter end: `sales - |redeem|` has a median absolute residual of 0.0012 and 84.1% of
fund-quarters within 1%, against 0.0024 and 67.4% for `sales - redeem` and 0.0790
for `sales + redeem`. The denominator is beginning-of-month TNA, reconstructed backward
through the quarter, not the quarter-end figure.

SPONSOR x MONTH compares claim categories within the same fund family and month.
A month-only specification is reported alongside it. These are conditional associations.

INFERENCE. Claim paths are reassigned to funds observed in all the required months,
holding calendar timing fixed. The script reports the current group sizes, cluster-based
standard errors and permutation p-values; it does not assume random actual claim adoption.

The reported contrast is rung 2 minus rung 5: process claims minus thematic claims
confirmed by the company-exposure holdings measure used here.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from src import paths as tp  # noqa: E402
import io, os, sys

import numpy as np
import pandas as pd

OUT = tp.out('analysis', 'claim_rungs', 'out')
LOGS = tp.out('analysis', 'claim_rungs', 'logs')
os.makedirs(OUT, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)

from src.analysis.flows.estimation import ols_cluster, demean, MIN_TNA        # noqa: E402
from src.analysis.flows.estimation import absorb, make_dims, absorbed_k                     # noqa: E402
from src.analysis.flows.estimation import OVERLAY                               # noqa: E402

LAD = tp.out('analysis', 'flow_ladder', 'out', 'fund_month_ladder.parquet')
THEME = tp.out('holdings', 'theme_universe.parquet')

# Every observed nonzero rung has its own dummy: only rung 0 is the reference group.
RUNGS = [1, 2, 3, 4, 5]
RUNG_NAME = {1: 'AI name only', 2: 'uses AI (process)', 3: 'theme, nothing named',
             4: 'theme, named NOT confirmed', 5: 'theme, named and CONFIRMED'}
PATH_NAME = {0: 'no claim', 1: 'name only', 2: 'process', 3: 'nothing named',
             4: 'not confirmed', 5: 'confirmed'}
# With B successful draws the minimum Monte Carlo p-value is 1/(B+1).
# A value at that floor means no draw exceeded the observed statistic, not greater precision.
NPERM = 2000
SEED = 20260918


def rung_dummies(rungs):
    """Encode all known nonzero rungs; reject unknown labels instead of using the control."""
    if not rungs.isin([0] + RUNGS).all():
        raise ValueError('The flow sample must contain only known rungs 0 through 5.')
    return pd.DataFrame({'r%d' % r: (rungs == r).astype(float) for r in RUNGS},
                        index=rungs.index)


def flow_groups(rungs):
    """Keep all six claim categories separate in the descriptive annual table."""
    groups = rungs.map(PATH_NAME)
    if groups.isna().any():
        raise ValueError('Cannot label an unknown rung as a no-claim fund.')
    return groups


def calendar_flow_lag(frame):
    """Previous calendar month's flow in a fund/month-sorted estimation panel.

    After missing-data filters, the previous row can be several months old. Such a row
    does not supply a one-month lag and is left missing.
    """
    month_number = frame.month.dt.year * 12 + frame.month.dt.month
    gap = month_number - month_number.groupby(frame.series_id, sort=False).shift()
    previous = frame.groupby('series_id', sort=False).nf.shift()
    return previous.where(gap.eq(1))


def latest_disclosed_holdings(frame, exposure_columns):
    """Resolve same-day filings by latest report date, rejecting conflicting exact keys."""
    frame = frame.copy()
    for column in ('filing_date', 'report_date'):
        frame[column] = pd.to_datetime(frame[column])
    keys = ['series_id', 'filing_date', 'report_date']
    counts = frame.groupby(keys, dropna=False)[exposure_columns].nunique(dropna=False)
    if counts.gt(1).any(axis=1).any():
        raise ValueError('Conflicting exposure values for the same fund, filing and report date.')
    return (frame.sort_values(['filing_date', 'series_id', 'report_date'],
                              kind='mergesort', na_position='first')
            .drop_duplicates(['series_id', 'filing_date'], keep='last'))


def sample_metadata(frame):
    """Observation/fund counts and date range for a coefficient table."""
    return {'n': len(frame), 'funds': frame.series_id.nunique(),
            'month_start': frame.month.min().date().isoformat(),
            'month_end': frame.month.max().date().isoformat()}


def panel(lad=LAD):
    """The estimation sample of `main`, built the same way, for the sensitivity analyses."""
    A = pd.read_parquet(lad)
    A['month'] = pd.to_datetime(A.month)
    born = A.groupby('series_id').month.min()
    A['age_m'] = ((A.month.dt.year - A.series_id.map(born).dt.year) * 12
                  + (A.month.dt.month - A.series_id.map(born).dt.month))
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
    RD = rung_dummies(D.rung)
    D[RD.columns] = RD
    return D


def reported_rungs(X, b, se):
    """Mask rung columns with no variation after fixed-effect absorption.

    This checks zero columns only; it is not a general collinearity diagnostic.
    Keep the fitted arrays unchanged for contrasts and permutation calculations.
    """
    identified = np.array([not np.allclose(X[:, i], 0) for i in range(len(RUNGS))])
    shown_b, shown_se = b.copy(), se.copy()
    shown_b[:len(RUNGS)] = np.where(identified, b[:len(RUNGS)], np.nan)
    shown_se[:len(RUNGS)] = np.where(identified, se[:len(RUNGS)], np.nan)
    return identified, shown_b, shown_se


def main():
    buf = io.StringIO()

    def w(s=''):
        buf.write(s + '\n')
        try:
            print(s)
        except UnicodeEncodeError:
            print(s.encode('ascii', 'replace').decode())

    A = pd.read_parquet(LAD)
    A['month'] = pd.to_datetime(A.month)
    born = A.groupby('series_id').month.min()
    A['age_m'] = ((A.month.dt.year - A.series_id.map(born).dt.year) * 12
                  + (A.month.dt.month - A.series_id.map(born).dt.month))
    import warnings
    warnings.filterwarnings('ignore', message='This pattern is interpreted')
    A['is_overlay'] = A.series_name.fillna('').str.contains(OVERLAY, regex=True)

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
    RD = rung_dummies(D.rung)
    D[RD.columns] = RD

    w('=' * 96)
    w('THE FLOW PREMIUM ALONG THE VERIFIABILITY LADDER')
    w('=' * 96)
    w('  estimation sample %d fund-months, %d funds, %d sponsors, %s .. %s'
      % (len(D), D.series_id.nunique(), D.sponsor.nunique(),
         D.month.min().date(), D.month.max().date()))
    w('  omitted category: rung 0 only; name-only funds have a separate rung 1 coefficient')
    w('  %-6s %-30s %12s %8s' % ('rung', 'meaning', 'fund-months', 'funds'))
    for r in [0] + RUNGS:
        g = D[D.rung == r]
        w('  %-6d %-30s %12d %8d'
          % (r, RUNG_NAME.get(r, 'no claim'), len(g), g.series_id.nunique()))
    w()

    XC = ['lsize', 'ret', 'nf_lag_w', 'lage']
    XN = ['r%d' % r for r in RUNGS] + XC
    rows = []
    store = {}
    for nm, fes in (('sponsor x month', [['sponsor', 'month']]),
                    ('month only (additive)', [['month']])):
        sub, dims = make_dims(D, fes)
        Y = absorb(sub[['nf_w']].values, dims)[:, 0]
        X = absorb(sub[XN].values, dims)
        b, se, V, G = ols_cluster(Y, X, sub.series_id.values,
                                  n_absorbed=absorbed_k(dims, sub.series_id.values))
        identified, shown_b, shown_se = reported_rungs(X, b, se)
        store[nm] = dict(b=b, se=se, V=V, sub=sub, dims=dims, Y=Y,
                         identified=identified)
        w('-' * 96)
        w('%s   (%d fund-months, %d clusters)' % (nm.upper(), len(sub), G))
        w('-' * 96)
        w('  %-6s %-30s %12s %9s %8s' % ('rung', 'meaning', 'coef pp/mo', 'se', 't'))
        for i, r in enumerate(RUNGS):
            w('  %-6d %-30s %12.3f %9.3f %8.2f'
              % (r, RUNG_NAME[r], shown_b[i], shown_se[i],
                 shown_b[i] / shown_se[i] if shown_se[i] else np.nan))
            rows.append({'spec': nm, 'rung': r, 'coef': shown_b[i], 'se': shown_se[i],
                         **sample_metadata(sub), 'identified': identified[i]})
        Lv = np.zeros(len(XN))
        Lv[RUNGS.index(2)] = 1.0
        Lv[RUNGS.index(5)] = -1.0
        c = float(Lv @ b)
        sc = float(np.sqrt(Lv @ V @ Lv))
        contrast_identified = bool(identified[RUNGS.index(2)] and identified[RUNGS.index(5)])
        if not contrast_identified:
            c = sc = np.nan
        w()
        w('  contrast  rung 2 (process claim) - rung 5 (holdings-confirmed theme):')
        w('     %+.3f pp/month   se %.3f   t %.2f' % (c, sc, c / sc if sc else np.nan))
        rows.append({'spec': nm, 'rung': -25, 'coef': c, 'se': sc, **sample_metadata(sub),
                     'identified': contrast_identified})
        w()

    # ---------------- robustness ----------------
    w('-' * 96)
    w('ROBUSTNESS')
    w('-' * 96)
    TH = pd.read_parquet(THEME)
    thematic = set(TH.loc[TH.is_thematic, 'series_id'])
    D['peer'] = D.series_id.isin(thematic)
    cuts = [('1 baseline', D),
            ('2 overlays excluded', D[~D.is_overlay]),
            ('3 age >= 12 months', D[D.age_m >= 12]),
            ('4 age >= 24 months', D[D.age_m >= 24]),
            ('5 thematic controls only', D[(D.rung != 0) | D.peer]),
            ('6 from 2022-12', D[D.month >= '2022-12-01']),
            ('7 through 2025', D[D.month < '2026-01-01'])]
    w('  %-26s %10s %10s %10s %12s'
      % ('cut', 'rung 2', 'rung 3', 'rung 5', 'r2 - r5 (t)'))
    for nm, s in cuts:
        sub, dims = make_dims(s, [['sponsor', 'month']])
        if len(sub) < 1000:
            continue
        Y = absorb(sub[['nf_w']].values, dims)[:, 0]
        X = absorb(sub[XN].values, dims)
        b, se, V, G = ols_cluster(Y, X, sub.series_id.values,
                                  n_absorbed=absorbed_k(dims, sub.series_id.values))
        identified, shown_b, shown_se = reported_rungs(X, b, se)
        Lv = np.zeros(len(XN)); Lv[RUNGS.index(2)] = 1; Lv[RUNGS.index(5)] = -1
        c = float(Lv @ b); sc = float(np.sqrt(Lv @ V @ Lv))
        contrast_identified = bool(identified[RUNGS.index(2)] and identified[RUNGS.index(5)])
        if not contrast_identified:
            c = sc = np.nan
        w('  %-26s %10.3f %10.3f %10.3f %6.2f (%4.2f)'
          % (nm, shown_b[RUNGS.index(2)], shown_b[RUNGS.index(3)], shown_b[RUNGS.index(5)], c,
             c / sc if sc else np.nan))
        for i, r in enumerate(RUNGS):
            rows.append({'spec': nm, 'rung': r, 'coef': shown_b[i], 'se': shown_se[i],
                         **sample_metadata(sub), 'identified': identified[i]})
        missing = [str(r) for i, r in enumerate(RUNGS) if not identified[i]]
        if missing:
            w('    no within-effect variation; coefficient unavailable for rung(s): ' + ', '.join(missing))
        rows.append({'spec': nm, 'rung': -25, 'coef': c, 'se': sc, **sample_metadata(sub),
                     'identified': contrast_identified})
    w()

    # ---------------- randomisation ----------------
    S = store['sponsor x month']
    sub = S['sub'].sort_values(['series_id', 'month']).copy()
    order = sub.index.values
    gid = pd.Series(S['dims'][0][0], index=S['sub'].index).loc[order].values
    ng = gid.max() + 1
    Yd = demean(sub[['nf_w']].values, gid, ng)[:, 0]
    Cd = demean(sub[XC].values, gid, ng)
    mi = ((sub.month.dt.year - 2019) * 12 + sub.month.dt.month).values
    sids = sub.series_id.values
    chg = np.flatnonzero(np.r_[True, sids[1:] != sids[:-1]])
    slc = {sids[a]: (a, b) for a, b in zip(chg, np.r_[chg[1:], len(sids)])}
    all_sid = np.array(list(slc.keys()))
    # Each treated fund's PATH moves, not its highest rung over its whole window: the
    # months it is at each rung, as observed. A recipient must be in the sample in every
    # one of those months.
    assert not sub.duplicated(['series_id', 'month']).any(), 'one row per fund-month'
    months = np.unique(mi)
    mpos = np.full(months.max() + 1, -1)
    mpos[months] = np.arange(len(months))
    alive = np.zeros((len(all_sid), len(months)), bool)
    for k_, s_ in enumerate(all_sid):
        a_, b_ = slc[s_]
        alive[k_, mpos[mi[a_:b_]]] = True
    tr = sub[sub.rung.isin(RUNGS)]
    jobs = []
    for s_, g_ in tr.groupby('series_id', sort=True):
        mm = ((g_.month.dt.year - 2019) * 12 + g_.month.dt.month).to_numpy()
        cols = np.array([RUNGS.index(int(r)) for r in g_.rung])
        jobs.append((s_, mm, cols, np.flatnonzero(alive[:, mpos[mm]].all(axis=1))))
    # the most constrained paths are placed first, so a draw is rarely left without a
    # recipient for a long path
    jobs.sort(key=lambda j: len(j[3]))
    # drawn onto itself, every path must reproduce the observed rung dummies exactly
    R0 = np.zeros((len(sub), len(RUNGS)))
    for s_, mm, cols, _ in jobs:
        a_, b_ = slc[s_]
        R0[a_ + np.searchsorted(mi[a_:b_], mm), cols] = 1.0
    obs_d = np.column_stack([(sub.rung.values == r).astype(float) for r in RUNGS])
    assert np.array_equal(R0, obs_d), 'self-assignment does not reproduce the rung dummies'
    rng = np.random.default_rng(SEED)
    obs = S['b'][:len(RUNGS)]
    obs_c = float(obs[RUNGS.index(2)] - obs[RUNGS.index(5)])
    hits = np.zeros(len(RUNGS)); hits_c = 0; drawn = 0
    for draw_index in range(NPERM):
        R = np.zeros((len(sub), len(RUNGS)))
        used, ok = set(), True
        for _s, mm, cols, pool in jobs:
            free = pool[~np.isin(pool, list(used))] if used else pool
            if not len(free):
                ok = False
                break
            j = int(free[rng.integers(len(free))])
            used.add(j)
            a, b_ = slc[all_sid[j]]
            R[a + np.searchsorted(mi[a:b_], mm), cols] = 1.0
        if not ok:
            continue
        Xp = np.hstack([demean(R, gid, ng), Cd])
        bp = np.linalg.pinv(Xp.T @ Xp) @ (Xp.T @ Yd)
        hits += np.abs(bp[:len(RUNGS)]) >= np.abs(obs)
        hits_c += abs(bp[RUNGS.index(2)] - bp[RUNGS.index(5)]) >= abs(obs_c)
        drawn += 1
        if (draw_index + 1) % 100 == 0:
            w('  randomisation progress: %d/%d attempts, %d successful draws'
              % (draw_index + 1, NPERM, drawn))

    w('-' * 96)
    w('RANDOMISATION INFERENCE  (%d draws; each treated fund\'s month-by-month claim path'
      % drawn)
    w('moved to a fund OBSERVED IN EVERY MONTH OF IT, so the calendar timing of AI claims and')
    w('each month\'s count at every rung are held fixed)')
    w('-' * 96)
    w('  %-6s %-30s %12s %14s' % ('rung', 'meaning', 'coef', 'randomisation p'))
    for i, r in enumerate(RUNGS):
        is_identified = bool(S['identified'][i])
        pv = (hits[i] + 1) / (drawn + 1) if is_identified else np.nan
        shown_obs = obs[i] if is_identified else np.nan
        w('  %-6d %-30s %12.3f %14.6f' % (r, RUNG_NAME[r], shown_obs, pv))
        rows.append({'spec': 'randomisation', 'rung': r, 'coef': shown_obs, 'se': np.nan,
                     'p': pv, 'n_draws': drawn, 'seed': SEED, **sample_metadata(sub),
                     'identified': is_identified})
    contrast_identified = bool(S['identified'][RUNGS.index(2)] and S['identified'][RUNGS.index(5)])
    pc = (hits_c + 1) / (drawn + 1) if contrast_identified else np.nan
    shown_c = obs_c if contrast_identified else np.nan
    w('  %-37s %12.3f %14.6f' % ('contrast rung 2 - rung 5', shown_c, pc))
    rows.append({'spec': 'randomisation', 'rung': -25, 'coef': shown_c, 'se': np.nan,
                 'p': pc, 'n_draws': drawn, 'seed': SEED, **sample_metadata(sub),
                 'identified': contrast_identified})
    w()

    # ---------------- time paths ----------------
    w('-' * 96)
    w('ALL CLAIM GROUPS  (mean within-month winsorised net flow, %, no controls)')
    w('-' * 96)
    D['grp'] = flow_groups(D.rung)
    T = (D.groupby([D.month.dt.year, 'grp']).nf_w.mean().unstack())
    cols = [PATH_NAME[r] for r in [0] + RUNGS if PATH_NAME[r] in T.columns]
    T = T.reindex(columns=cols)
    w('  %-8s' % 'year' + ''.join('%16s' % c for c in cols))
    for y_, row in T.iterrows():
        w('  %-8s' % y_ + ''.join('%16.3f' % row[c] for c in cols))
    w()
    w('  These are descriptive means; only rung 0 is labelled no claim.')
    w('  Name-only and named-but-unconfirmed claims remain separate throughout.')
    w()

    estimates = pd.DataFrame(rows)
    estimates.to_parquet(os.path.join(OUT, 'premium.parquet'), index=False)
    T.to_parquet(os.path.join(OUT, 'time_paths.parquet'))
    tables = tp.RESULTS / 'tables'
    tables.mkdir(parents=True, exist_ok=True)
    estimates.to_csv(tables / 'flow_contrast_premium.csv', index=False, float_format='%.12g')
    T.rename_axis('year').to_csv(tables / 'flow_contrast_paths.csv', float_format='%.12g')
    w('wrote out/premium.parquet, out/time_paths.parquet')

    with io.open(os.path.join(LOGS, 'claim_rungs.txt'), 'w', encoding='utf-8') as fh:
        fh.write(buf.getvalue())


if __name__ == '__main__':
    main()

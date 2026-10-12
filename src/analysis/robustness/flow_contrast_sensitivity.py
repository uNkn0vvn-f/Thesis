"""Sensitivity of the theme-with-holdings-support versus process-only flow contrast.

Builds the estimation panel with the flow regressions' own code (claim_rungs.panel), then
reports, all against the same baseline:

  * the baseline coefficients and the covariance-aware contrast;
  * group sizes, sponsors (registrant trusts) and the observations that identify the
    contrast under registrant x month fixed effects;
  * alternative clustering (registrant, month, two-way fund and month);
  * controls for past performance and flow persistence; age alternatives;
  * outlier handling, size floors, leave-one-fund-out and leave-one-registrant-out;
  * year-by-year estimates;
  * comparisons inside AI-theme claimants (confirmed against unconfirmed and against
    no role named), with and without portfolio AI exposure.

Coefficients are percentage points of monthly net flow (% of beginning-of-month assets).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src import paths as tp  # noqa: E402
from src.analysis.flows.estimation import absorb, make_dims, absorbed_k  # noqa: E402
from src.analysis.flows import claim_rungs  # noqa: E402
from src.analysis.flows.claim_rungs import RUNGS, latest_disclosed_holdings  # noqa: E402
from src.analysis.robustness.common import TABLES, INTERIM, LABELS, path  # noqa: E402

# rungs, holdings support and the AI-revenue and technology controls all come from the
# selected labels (THESIS_LABELS)
LADDER = str(path('ladder'))


def panel():
    return claim_rungs.panel(LADDER)

XC = ['lsize', 'ret', 'nf_lag_w', 'lage']
RN = ['r%d' % r for r in RUNGS]


def group_sum(M, codes):
    codes = pd.factorize(pd.Series(codes).astype(str).values)[0]
    G = codes.max() + 1
    out = np.zeros((G, M.shape[1]))
    np.add.at(out, codes, M)
    return out, G


def fit(D, xn, fes=(('sponsor', 'month'),), clusters=('series_id',)):
    ok = np.isfinite(D[list(xn) + ['nf_w']].to_numpy(dtype=float)).all(axis=1)
    D = D[ok]
    sub, dims = make_dims(D, [list(f) for f in fes])
    Y = absorb(sub[['nf_w']].values, dims)[:, 0]
    X = absorb(sub[xn].values.astype(float), dims)
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ (X.T @ Y)
    u = Y - X @ b
    N = len(Y)
    Vs = {}
    for cl in clusters:
        if cl == 'fund+month':
            parts = []
            for c in ('series_id', 'month', None):
                key = sub[c].values if c else np.arange(N)
                M, G = group_sum(X * u[:, None], key)
                K = X.shape[1] + absorbed_k(dims, key)
                parts.append(XtXi @ (M.T @ M) @ XtXi * (G / (G - 1)) * ((N - 1) / (N - K)))
            Vs[cl] = parts[0] + parts[1] - parts[2]
        else:
            M, G = group_sum(X * u[:, None], sub[cl].values)
            K = X.shape[1] + absorbed_k(dims, sub[cl].values)
            Vs[cl] = XtXi @ (M.T @ M) @ XtXi * (G / (G - 1)) * ((N - 1) / (N - K))
    return b, Vs, sub


def contrast(b, V, xn, plus, minus):
    L = np.zeros(len(xn))
    L[xn.index(plus)] = 1
    L[xn.index(minus)] = -1
    c = float(L @ b)
    return c, float(np.sqrt(max(L @ V @ L, 0)))


def row(name, b, V, sub, xn, extra=None):
    c, se = contrast(b, V, xn, 'r5', 'r2')
    rec = {'spec': name, 'r5_minus_r2': c, 'se': se, 't': c / se if se else np.nan,
           'ci_lo': c - 1.96 * se, 'ci_hi': c + 1.96 * se,
           'r5': b[xn.index('r5')], 'r5_se': np.sqrt(V[xn.index('r5'), xn.index('r5')]),
           'r2': b[xn.index('r2')], 'r2_se': np.sqrt(V[xn.index('r2'), xn.index('r2')]),
           'n': len(sub), 'funds': sub.series_id.nunique(),
           'r5_funds': int(sub.loc[sub.r5 == 1, 'series_id'].nunique()),
           'r5_fund_months': int((sub.r5 == 1).sum()),
           'r2_funds': int(sub.loc[sub.r2 == 1, 'series_id'].nunique()),
           'r2_fund_months': int((sub.r2 == 1).sum())}
    if extra:
        rec.update(extra)
    return rec


def past_returns(D):
    F = pd.read_parquet(INTERIM / 'analysis' / 'fund_flows' / 'out' / 'fund_month_flows.parquet',
                        columns=['series_id', 'month', 'ret', 'nf_beg'])
    F = F[F.series_id.notna() & (F.series_id != '')].copy()
    F['month'] = pd.to_datetime(F.month)
    F = F.sort_values(['series_id', 'month']).drop_duplicates(['series_id', 'month'])
    F['mi'] = F.month.dt.year * 12 + F.month.dt.month
    full = []
    for sid, g in F.groupby('series_id', sort=False):
        idx = pd.RangeIndex(g.mi.min(), g.mi.max() + 1)
        s = g.set_index('mi')
        lr = np.log1p(s.ret.reindex(idx) / 100)
        nf = s.nf_beg.reindex(idx).replace([np.inf, -np.inf], np.nan)
        out = pd.DataFrame({'mi': idx, 'series_id': sid,
                            'cum3': np.expm1(lr.shift(1).rolling(3, min_periods=3).sum()) * 100,
                            'cum12': np.expm1(lr.shift(1).rolling(12, min_periods=12).sum()) * 100,
                            'cum12_any': np.expm1(lr.shift(1).rolling(12, min_periods=1).sum()) * 100,
                            'nf_lag2': nf.shift(2), 'nf_lag3': nf.shift(3)})
        full.append(out)
    P = pd.concat(full, ignore_index=True)
    D = D.copy()
    D['mi'] = D.month.dt.year * 12 + D.month.dt.month
    return D.merge(P, on=['series_id', 'mi'], how='left')


def main():
    D = panel()
    log = {'labels': LABELS, 'ladder': tp.portable(LADDER),
           'controls_ai_rev_tech_from': '%s exposure table' % LABELS}
    xn = RN + XC
    rows = []

    b, Vs, sub = fit(D, xn, clusters=('series_id', 'sponsor', 'month', 'fund+month'))
    for cl in ('series_id', 'sponsor', 'month', 'fund+month'):
        rows.append(row('baseline, cluster ' + cl, b, Vs[cl], sub, xn))
    V = Vs['series_id']
    log['replication'] = {'r5': b[xn.index('r5')], 'r2': b[xn.index('r2')],
                          'r2_minus_r5': -contrast(b, V, xn, 'r5', 'r2')[0],
                          'se': contrast(b, V, xn, 'r5', 'r2')[1], 'n': len(sub),
                          'funds': int(sub.series_id.nunique())}
    # internal contrasts with covariance
    log['contrasts_baseline'] = {}
    for p, m in (('r5', 'r2'), ('r5', 'r3'), ('r5', 'r4'), ('r3', 'r2'), ('r4', 'r3')):
        c, se = contrast(b, V, xn, p, m)
        log['contrasts_baseline'][p + '-' + m] = {'coef': c, 'se': se, 'ci': [c - 1.96 * se, c + 1.96 * se]}

    # ---- group sizes and identifying variation ----
    cnt = []
    for r in [0] + RUNGS:
        g = sub[sub.rung == r]
        cnt.append({'rung': r, 'fund_months': len(g), 'funds': g.series_id.nunique(),
                    'registrants': g.sponsor.nunique()})
    cell = sub.groupby(['sponsor', 'month']).rung.agg(lambda s: s.nunique() > 1)
    sub2 = sub.merge(cell.rename('cell_varies').reset_index(), on=['sponsor', 'month'])
    for c in cnt:
        g = sub2[sub2.rung == c['rung']]
        c['fund_months_in_cells_with_other_rungs'] = int(g.cell_varies.sum())
        c['funds_in_cells_with_other_rungs'] = int(g.loc[g.cell_varies, 'series_id'].nunique())
    pd.DataFrame(cnt).to_csv(TABLES / 'flow_contrast_group_sizes.csv', index=False)
    log['group_sizes'] = cnt
    r5 = sub[sub.rung == 5]
    top = (r5.groupby('series_id').agg(name=('series_name', 'last'), registrant=('registrant_name', 'last'),
                                       fund_months=('month', 'size'), mean_flow=('nf_w', 'mean'),
                                       first=('month', 'min'), last=('month', 'max'))
           .sort_values('fund_months', ascending=False))
    top.to_csv(TABLES / 'flow_contrast_rung5_funds.csv')
    r2 = sub[sub.rung == 2]
    r2.groupby('series_id').agg(name=('series_name', 'last'), registrant=('registrant_name', 'last'),
                                fund_months=('month', 'size'), mean_flow=('nf_w', 'mean')).sort_values(
        'fund_months', ascending=False).to_csv(TABLES / 'flow_contrast_rung2_funds.csv')
    log['rung5_registrants_top'] = r5.registrant_name.value_counts().head(8).to_dict()
    log['rung2_registrants_top'] = r2.registrant_name.value_counts().head(8).to_dict()

    # ---- age variable ----
    A0 = pd.read_parquet(LADDER, columns=['series_id', 'month'])
    born = pd.to_datetime(A0.groupby('series_id').month.min())
    # left-censored: first flow month no later than the first estimation month (as flow_contrast_age.py)
    D['left_censored'] = D.series_id.map(born) <= pd.Timestamp('2020-03-01')
    log['age'] = {'definition': 'months since the first month the fund appears in the flow panel (panel starts %s); '
                                'left-censored if that month is 2020-03 or earlier' % born.min().date(),
                  'share_fund_months_left_censored': float(D.left_censored.mean()),
                  'r5_share_left_censored': float(D.loc[D.r5 == 1, 'left_censored'].mean()),
                  'r2_share_left_censored': float(D.loc[D.r2 == 1, 'left_censored'].mean()),
                  'r5_median_age_months': float(D.loc[D.r5 == 1, 'age_m'].median()),
                  'r2_median_age_months': float(D.loc[D.r2 == 1, 'age_m'].median())}
    bins = pd.cut(D.age_m, [-1, 6, 12, 24, 36, 60, 1000], labels=False)
    for k in range(6):
        D['ab%d' % k] = ((bins == k) & ~D.left_censored).astype(float)
    D['lc'] = D.left_censored.astype(float)
    xa = RN + ['lsize', 'ret', 'nf_lag_w', 'lc'] + ['ab%d' % k for k in range(1, 6)]
    b2, V2, s2 = fit(D, xa)
    rows.append(row('age bins + left-censoring indicator (no log age)', b2, V2['series_id'], s2, xa))
    newf = D[~D.left_censored]
    b2, V2, s2 = fit(newf, xn)
    rows.append(row('funds first observed after panel start only', b2, V2['series_id'], s2, xn))

    # ---- past performance and flow persistence ----
    D = past_returns(D)
    for c in ('cum3', 'cum12', 'nf_lag2', 'nf_lag3'):
        lo = D.groupby('month')[c].transform(lambda s: s.quantile(.01))
        hi = D.groupby('month')[c].transform(lambda s: s.quantile(.99))
        D[c + '_w'] = D[c].clip(lo, hi)
        D[c + '_miss'] = D[c].isna().astype(float)
        D[c + '_f'] = D[c + '_w'].fillna(0)
    xp = xn + ['cum3_f', 'cum3_miss', 'cum12_f', 'cum12_miss']
    b2, V2, s2 = fit(D, xp)
    rows.append(row('+ past 3- and 12-month returns (missing = indicator)', b2, V2['series_id'], s2, xp))
    full12 = D[D.cum12.notna()]
    xp2 = xn + ['cum3_w', 'cum12_w']
    b2, V2, s2 = fit(full12, xp2)
    rows.append(row('+ past returns, funds with 12 months of history', b2, V2['series_id'], s2, xp2))
    xf = xp + ['nf_lag2_f', 'nf_lag2_miss', 'nf_lag3_f', 'nf_lag3_miss']
    b2, V2, s2 = fit(D, xf)
    rows.append(row('+ past returns + flow lags 2 and 3', b2, V2['series_id'], s2, xf))

    # ---- outliers and size ----
    for lo_q, hi_q in ((.025, .975), (.05, .95)):
        E = D.copy()
        for c in ('nf', 'nf_lag'):
            lo = E.groupby('month')[c].transform(lambda s: s.quantile(lo_q))
            hi = E.groupby('month')[c].transform(lambda s: s.quantile(hi_q))
            E[c + '_w'] = E[c].clip(lo, hi)
        b2, V2, s2 = fit(E, xn)
        rows.append(row('winsorise %.1f/%.1f within month' % (100 * lo_q, 100 * hi_q), b2, V2['series_id'], s2, xn))
    b2, V2, s2 = fit(D[D.nf.abs() <= 20], xn)
    rows.append(row('drop |raw flow| > 20% of assets', b2, V2['series_id'], s2, xn))
    for floor in (10e6, 50e6):
        b2, V2, s2 = fit(D[D.na_beg_recon >= floor], xn)
        rows.append(row('assets >= $%dm' % (floor / 1e6), b2, V2['series_id'], s2, xn))

    # ---- years ----
    for y in range(2020, 2027):
        Dy = D[D.month.dt.year == y]
        if (Dy.r5 == 1).sum() < 20:
            continue
        try:
            b2, V2, s2 = fit(Dy, xn)
            rows.append(row('year %d' % y, b2, V2['series_id'], s2, xn))
        except Exception as exc:  # noqa: BLE001
            log.setdefault('year_failures', {})[y] = str(exc)

    # ---- leave one out ----
    loo = []
    for f in sorted(D.loc[D.r5 == 1, 'series_id'].unique()):
        b2, V2, s2 = fit(D[D.series_id != f], xn)
        c, se = contrast(b2, V2['series_id'], xn, 'r5', 'r2')
        loo.append({'dropped': 'fund ' + f, 'r5_minus_r2': c, 'se': se})
    for f in sorted(D.loc[D.r5 == 1, 'sponsor'].unique()):
        b2, V2, s2 = fit(D[D.sponsor != f], xn)
        c, se = contrast(b2, V2['series_id'], xn, 'r5', 'r2')
        loo.append({'dropped': 'registrant ' + f, 'r5_minus_r2': c, 'se': se,
                    'registrant_name': D.loc[D.sponsor == f, 'registrant_name'].dropna().iloc[-1]
                    if D.loc[D.sponsor == f, 'registrant_name'].notna().any() else ''})
    for f in sorted(D.loc[D.r2 == 1, 'sponsor'].unique()):
        b2, V2, s2 = fit(D[D.sponsor != f], xn)
        c, se = contrast(b2, V2['series_id'], xn, 'r5', 'r2')
        loo.append({'dropped': 'process registrant ' + f, 'r5_minus_r2': c, 'se': se})
    LOO = pd.DataFrame(loo)
    LOO.to_csv(TABLES / 'flow_contrast_leave_one_out.csv', index=False)
    for k in ('fund', 'registrant', 'process registrant'):
        g = LOO[LOO.dropped.str.startswith(k + ' ')]
        log['leave_one_out_' + k.replace(' ', '_')] = {'n': len(g), 'min': float(g.r5_minus_r2.min()),
                                                       'max': float(g.r5_minus_r2.max()),
                                                       'at_min': g.loc[g.r5_minus_r2.idxmin(), 'dropped']}
    # drop the five rung-5 funds with the most fund-months together
    big = top.index[:5].tolist()
    b2, V2, s2 = fit(D[~D.series_id.isin(big)], xn)
    rows.append(row('drop the 5 rung-5 funds with most fund-months', b2, V2['series_id'], s2, xn,
                    {'dropped': ','.join(big)}))

    # ---- inside AI-theme claimants, with exposure ----
    E = pd.read_parquet(path('exposure'),
                        columns=['series_id', 'report_date', 'filing_date', 'labelled', 'ai_rev',
                                 'tech', 'abn_ai_rev', 'developer', 'enabler', 'adopter'])
    E['report_date'] = pd.to_datetime(E.report_date)
    E['filing_date'] = pd.to_datetime(E.filing_date)
    E = E[E.labelled >= 50].dropna(subset=['filing_date'])
    E = latest_disclosed_holdings(E, ['ai_rev', 'tech', 'abn_ai_rev'])
    T = D.sort_values('month')
    T = pd.merge_asof(T, E.sort_values('filing_date'), left_on='month', right_on='filing_date',
                      by='series_id', direction='backward', allow_exact_matches=False)
    T['fresh'] = T.report_date.notna() & (T.month - T.report_date).dt.days.between(0, 366)
    TH = T[T.rung.isin([3, 4, 5])].copy()
    THx = TH[TH.fresh].copy()
    for c in ('r4', 'r5'):
        THx[c] = THx[c].astype(float)
    inside = []
    for fe_name, fes in (('month FE', (('month',),)), ('registrant x month FE', (('sponsor', 'month'),))):
        for spec, xs in (('confirmed and unconfirmed vs nothing named', ['r4', 'r5'] + XC),
                         ('+ AI revenue share and technology share', ['r4', 'r5'] + XC + ['ai_rev', 'tech']),
                         ('+ past returns', ['r4', 'r5'] + XC + ['ai_rev', 'tech', 'cum3_f', 'cum3_miss', 'cum12_f', 'cum12_miss'])):
            try:
                bb, VV, ss = fit(THx, xs, fes=fes)
                V0 = VV['series_id']
                rec = {'fe': fe_name, 'spec': spec, 'n': len(ss), 'funds': ss.series_id.nunique(),
                       'r5_vs_r3': bb[xs.index('r5')], 'r5_vs_r3_se': np.sqrt(V0[xs.index('r5'), xs.index('r5')]),
                       'r5_funds': int(ss.loc[ss.r5 == 1, 'series_id'].nunique()),
                       'r4_funds': int(ss.loc[ss.r4 == 1, 'series_id'].nunique()),
                       'r3_funds': int(ss.loc[ss.rung == 3, 'series_id'].nunique())}
                c, se = contrast(bb, V0, xs, 'r5', 'r4')
                rec.update(r5_vs_r4=c, r5_vs_r4_se=se)
                inside.append(rec)
            except Exception as exc:  # noqa: BLE001
                inside.append({'fe': fe_name, 'spec': spec, 'error': str(exc)})
    pd.DataFrame(inside).to_csv(TABLES / 'flow_contrast_inside_theme.csv', index=False)
    log['inside_theme'] = inside
    log['theme_fund_months_with_fresh_holdings'] = {'rung3': int((THx.rung == 3).sum()), 'rung4': int((THx.rung == 4).sum()),
                                                    'rung5': int((THx.rung == 5).sum()),
                                                    'rung3_without_fresh_holdings': int(((TH.rung == 3) & ~TH.fresh).sum())}

    # ---- process vs theme-with-support, conditional on AI exposure (funds with holdings) ----
    P = T[T.fresh & T.rung.isin([0, 2, 3, 4, 5])].copy()
    for spec, xs in (('holdings-covered sample', xn), ('+ AI revenue share and technology share', xn + ['ai_rev', 'tech']),
                     ('+ exposures + past returns', xn + ['ai_rev', 'tech', 'cum3_f', 'cum3_miss', 'cum12_f', 'cum12_miss'])):
        bb, VV, ss = fit(P, xs)
        rows.append(row('exposure-covered: ' + spec, bb, VV['series_id'], ss, xs))

    R = pd.DataFrame(rows)
    R.to_csv(TABLES / 'flow_contrast_sensitivity.csv', index=False, float_format='%.6g')
    (TABLES / 'flow_contrast_log.json').write_text(json.dumps(log, indent=2, default=str), encoding='utf-8')
    pd.set_option('display.width', 250)
    print(R[['spec', 'r5_minus_r2', 'se', 'ci_lo', 'ci_hi', 'n', 'funds', 'r5_funds', 'r2_funds']].round(3).to_string())
    print(json.dumps(log, indent=2, default=str))


if __name__ == '__main__':
    main()

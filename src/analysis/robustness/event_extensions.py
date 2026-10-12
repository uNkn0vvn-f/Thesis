"""Event-analysis extensions: where the pre-event differences come from, and what the design can carry.

All specifications below were fixed before estimation and are reported in full:

  A  the single-event and stacked estimates;
  B  event-time coefficients re-expressed relative to the pre-period mean, with the
     per-standard-deviation magnitude;
  C  exposure quintiles: which part of the exposure distribution carries the estimate;
  D  composition: balanced windows, thematic versus other funds, AI claimants;
  E  leave-one-fund-out influence;
  F  event-month handling: event month in pre (baseline), dropped, coded post; +/-6-month
     windows; event month and first post month dropped;
  G  past 3- and 12-month returns and flow lags 2-3 as controls (performance chasing and
     flow persistence; these controls can absorb part of a genuine response to the news);
  H  finer industry composition of the pre-event portfolio (semiconductors, software and
     IT services, computer hardware, other electronics, communications services), each
     interacted with post and post x sign, mapped from the same company universe SIC
     codes as the exposure builder;
  I  the same designs with the annual label vintages instead of the pre-event labels;
  J  every candidate event month from 2021-04 to 2025-06 with date-specific information
     (annual label vintages, portfolios filed before that month): where do the actual
     dates rank?
  K  fund-level collapse: post-minus-pre mean flow on pre-event exposure.

Pre-event and annual labels follow THESIS_LABELS (common.py).

Coefficients: percentage points of monthly net flow per percentage point of pre-event
commercial AI-revenue exposure.
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

from src.analysis.robustness.common import TABLES, INTERIM, PROCESSED, LABELS, path, preevent_path  # noqa: E402

# pre-event exposures (A-H, K) and annual-vintage exposures (I, J) both follow THESIS_LABELS
EXP_PRE = {t: str(preevent_path(t)) for t in ('chatgpt', 'deepseek')}
EXP_MAIN = str(path('exposure'))
from src.analysis.events import two_events as S2  # noqa: E402
from src.analysis.events.event_dates import treat_cutoff  # noqa: E402
from src.analysis.flows.estimation import absorb, make_dims, absorbed_k  # noqa: E402
from src.analysis.robustness.flow_contrast_sensitivity import past_returns, group_sum  # noqa: E402

XC = ['lsize', 'ret', 'nf_lag_w']
EVENTS = [('chatgpt', pd.Timestamp('2022-12-01'), 1.0), ('deepseek', pd.Timestamp('2025-02-01'), -1.0)]


def fit(D, xn, fes):
    sub, dims = make_dims(D, [list(f) for f in fes])
    Y = absorb(sub[['nf_w']].values, dims)[:, 0]
    X = absorb(sub[xn].values.astype(float), dims)
    keep = X.std(axis=0) > 1e-12
    if not keep.all():
        raise ValueError('no variation in ' + ', '.join(np.array(xn)[~keep]))
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ (X.T @ Y)
    u = Y - X @ b
    M, G = group_sum(X * u[:, None], sub.series_id.values)
    N, K = len(Y), X.shape[1] + absorbed_k(dims, sub.series_id.values)
    V = XtXi @ (M.T @ M) @ XtXi * (G / (G - 1)) * ((N - 1) / (N - K))
    return b, V, sub


def single(D, extra=(), treat='ai_pre'):
    D = D.copy()
    D['ai_post'] = D[treat] * D.post
    D['cl_post'] = D.claim_pre * D.post
    xn = ['ai_post', 'cl_post'] + XC + list(extra)
    b, V, sub = fit(D, xn, (('series_id',), ('month',)))
    return b[0], float(np.sqrt(V[0, 0])), len(sub), sub.series_id.nunique()


def stacked(D, extra=()):
    D = D.copy()
    D['ai_post'] = D.ai_pre * D.post
    D['ai_post_sign'] = D.ai_post * D.sign
    D['cl_post'] = D.claim_pre * D.post
    D['cl_post_sign'] = D.cl_post * D.sign
    xn = ['ai_post', 'ai_post_sign', 'cl_post', 'cl_post_sign'] + XC + list(extra)
    b, V, sub = fit(D, xn, (('uid',), ('win', 'month')))
    return b[1], float(np.sqrt(V[1, 1])), len(sub), sub.series_id.nunique(), b[0], float(np.sqrt(V[0, 0]))


def add_k(D, shock):
    D = D.copy()
    D['k'] = (D.month.dt.year - shock.year) * 12 + D.month.dt.month - shock.month
    return D


def event_study(D, treat='ai_pre'):
    ks = sorted(D.k.unique())
    cols = []
    D = D.copy()
    for k in ks:
        if k != -1:
            D['x%d' % k] = D[treat] * (D.k == k)
            cols.append('x%d' % k)
    b, V, sub = fit(D, cols + XC, (('series_id',), ('month',)))
    full = {k: (b[cols.index('x%d' % k)] if k != -1 else 0.0) for k in ks}
    pre = [k for k in ks if k < 0]
    # relative to the pre-period mean (the reference month counts as zero)
    rows = []
    for k in ks:
        L = np.zeros(len(cols) + len(XC))
        if k != -1:
            L[cols.index('x%d' % k)] += 1
        for p in pre:
            if p != -1:
                L[cols.index('x%d' % p)] -= 1 / len(pre)
        rows.append({'k': k, 'coef_ref_k-1': full[k], 'coef_rel_pre_mean': float(L @ b),
                     'se_rel_pre_mean': float(np.sqrt(L @ V @ L))})
    return pd.DataFrame(rows)


def pre_portfolio(exp, shock):
    E = pd.read_parquet(exp)
    E['report_date'] = pd.to_datetime(E.report_date)
    E['filing_date'] = pd.to_datetime(E.filing_date)
    cut = treat_cutoff(shock)
    pre = E[(E.filing_date < cut) & (E.labelled >= 50)]
    return pre.sort_values('report_date').groupby('series_id').tail(1)[['series_id', 'q', 'report_date']]


def industry_shares():
    H = pd.read_csv(PROCESSED / 'holdings_for_exposure.csv.gz',
                    dtype={'series_id': str, 'gvkey': str})
    H = H[H.value.notna() & (H.value > 0) & H.gvkey.notna()]
    U = pd.read_parquet(INTERIM / 'firm_labels' / 'firm_universe.parquet', columns=['gvkey', 'sic4', 'sector'])
    U['gvkey'] = U.gvkey.astype(str)
    H = H.merge(U, on='gvkey', how='inner')            # the labelled universe, as ai_exposure
    s = pd.to_numeric(H.sic4, errors='coerce')
    groups = {'ind_semis': s.eq(3674), 'ind_software': s.between(7370, 7379),
              'ind_hardware': s.between(3570, 3579),
              'ind_electronics': s.between(3600, 3699) & ~s.eq(3674),
              'ind_commsvc': s.between(4800, 4899)}
    tot = H.groupby(['series_id', 'q']).value.sum()
    out = pd.DataFrame(index=tot.index)
    for k, m in groups.items():
        out[k] = 100 * H[m].groupby(['series_id', 'q']).value.sum().reindex(tot.index).fillna(0) / tot
    return out.reset_index()


def build_windows(L, exp_map, win=12):
    S2.WIN = win
    parts = []
    for name, shock, sign in EVENTS:
        D = S2.window(name, shock, sign, exp_map[name], L)
        parts.append(add_k(D, shock))
    return parts


def main():
    log = {'labels': LABELS}
    L = S2.panel()
    L = past_returns(L.rename(columns={}))
    for c in ('cum3', 'cum12', 'nf_lag2', 'nf_lag3'):
        lo = L.groupby('month')[c].transform(lambda s: s.quantile(.01))
        hi = L.groupby('month')[c].transform(lambda s: s.quantile(.99))
        L[c + '_f'] = L[c].clip(lo, hi).fillna(0)
        L[c + '_miss'] = L[c].isna().astype(float)
    PR = ['cum3_f', 'cum3_miss', 'cum12_f', 'cum12_miss']
    FL = ['nf_lag2_f', 'nf_lag2_miss', 'nf_lag3_f', 'nf_lag3_miss']
    TH = pd.read_parquet(INTERIM / 'holdings' / 'theme_universe.parquet')
    thematic = set(TH.loc[TH.is_thematic, 'series_id'])

    res = []

    def rec(group, spec, event, out):
        if event == 'stacked':
            b, se, n, g, a, ase = out
            res.append({'group': group, 'spec': spec, 'event': event, 'coef': b, 'se': se, 't': b / se,
                        'n': n, 'funds': g, 'standing_a': a, 'standing_a_se': ase})
        else:
            b, se, n, g = out
            res.append({'group': group, 'spec': spec, 'event': event, 'coef': b, 'se': se, 't': b / se,
                        'n': n, 'funds': g})

    parts = build_windows(L, EXP_PRE)
    allD = pd.concat(parts, ignore_index=True)

    # A replication
    for D, (name, shock, sign) in zip(parts, EVENTS):
        rec('A replication', 'baseline', name, single(D))
    rec('A replication', 'baseline', 'stacked', stacked(allD))

    # B event time relative to pre mean
    es_rows = []
    for D, (name, shock, sign) in zip(parts, EVENTS):
        T = event_study(D)
        sd = D.groupby('series_id').ai_pre.first().std()
        T['event'] = name
        T['per_sd_rel_pre_mean'] = T.coef_rel_pre_mean * sd
        T['exposure_sd'] = sd
        es_rows.append(T)
        pre = T[T.k < 0]
        post = T[T.k >= 0]
        log['event_time_' + name] = {
            'exposure_sd_pp': float(sd),
            'pre_rel_mean_min': float(pre.coef_rel_pre_mean.min()), 'pre_rel_mean_max': float(pre.coef_rel_pre_mean.max()),
            'post_rel_mean_avg': float(post.coef_rel_pre_mean.mean()),
            'pre_month_range_per_sd_pp': float(sd * (pre.coef_rel_pre_mean.max() - pre.coef_rel_pre_mean.min())),
            'did_per_sd_pp': float(sd * single(D)[0]),
            'reference_month_k-1_rel_pre_mean': float(T.loc[T.k == -1, 'coef_rel_pre_mean'].iloc[0]),
            'post_peak_k': int(post.loc[post.coef_rel_pre_mean.abs().idxmax(), 'k'])}
    ES = pd.concat(es_rows, ignore_index=True)
    ES.to_csv(TABLES / 'event_time.csv', index=False, float_format='%.6g')

    # C exposure quintiles
    qrows = []
    for D, (name, shock, sign) in zip(parts, EVENTS):
        f = D.groupby('series_id').ai_pre.first()
        qs = pd.qcut(f.rank(method='first'), 5, labels=False)
        D = D.copy()
        D['qb'] = D.series_id.map(qs)
        for k in range(1, 5):
            D['q%d' % (k + 1)] = (D.qb == k).astype(float)
        D['cl_post'] = D.claim_pre * D.post
        xs = []
        for k in range(2, 6):
            D['q%d_post' % k] = D['q%d' % k] * D.post
            xs.append('q%d_post' % k)
        b, V, sub = fit(D, xs + ['cl_post'] + XC, (('series_id',), ('month',)))
        for i, k in enumerate(range(2, 6)):
            qrows.append({'event': name, 'quintile': k, 'coef_vs_q1': b[i], 'se': float(np.sqrt(V[i, i])),
                          'mean_exposure': float(f[qs == k - 1].mean()), 'funds': int((qs == k - 1).sum())})
        qrows.append({'event': name, 'quintile': 1, 'coef_vs_q1': 0.0, 'se': np.nan,
                      'mean_exposure': float(f[qs == 0].mean()), 'funds': int((qs == 0).sum())})
        # top-quintile event study, pre-period only: Wald on pre months
        D['top'] = (D.qb == 4).astype(float)
        T = event_study(D, treat='top')
        T['event'] = name
        T.to_csv(TABLES / ('event_time_topquintile_%s.csv' % name), index=False, float_format='%.6g')
    pd.DataFrame(qrows).sort_values(['event', 'quintile']).to_csv(TABLES / 'event_quintiles.csv', index=False, float_format='%.6g')

    # D composition
    for D, (name, shock, sign) in zip(parts, EVENTS):
        nmon = D.groupby('series_id').month.nunique()
        bal = D[D.series_id.map(nmon) == D.k.nunique()]
        rec('D composition', 'balanced window', name, single(bal))
        rec('D composition', 'thematic universe', name, single(D[D.series_id.isin(thematic)]))
        rec('D composition', 'non-thematic funds', name, single(D[~D.series_id.isin(thematic)]))
        nc = D[D.claim_pre == 0].copy()
        nc['claim_pre'] = 0.0
        nc2 = nc.copy()
        nc2['ai_post'] = nc2.ai_pre * nc2.post
        b, V, sub = fit(nc2, ['ai_post'] + XC, (('series_id',), ('month',)))
        res.append({'group': 'D composition', 'spec': 'funds without a pre-event AI claim', 'event': name,
                    'coef': b[0], 'se': float(np.sqrt(V[0, 0])), 't': b[0] / np.sqrt(V[0, 0]),
                    'n': len(sub), 'funds': sub.series_id.nunique()})
    nmon = allD.groupby('uid').month.nunique()
    rec('D composition', 'balanced windows', 'stacked', stacked(allD[allD.uid.map(nmon) == 24]))
    both = set(parts[0].series_id) & set(parts[1].series_id)
    rec('D composition', 'funds in both windows', 'stacked', stacked(allD[allD.series_id.isin(both)]))

    # E leave-one-fund-out
    infl = []
    for D, (name, shock, sign) in zip(parts, EVENTS):
        base = single(D)[0]
        for f in D.series_id.unique():
            b = single(D[D.series_id != f])[0]
            infl.append({'event': name, 'series_id': f, 'coef_without': b, 'change': b - base,
                         'ai_pre': float(D.loc[D.series_id == f, 'ai_pre'].iloc[0])})
    INF = pd.DataFrame(infl)
    nm = pd.read_parquet(INTERIM / 'holdings' / 'holdings_bridged.parquet', columns=['series_id', 'series_name']) \
        .drop_duplicates('series_id').set_index('series_id').series_name
    INF['name'] = INF.series_id.map(nm)
    INF.to_csv(TABLES / 'event_leave_one_fund_out.csv', index=False, float_format='%.6g')
    for name in ('chatgpt', 'deepseek'):
        g = INF[INF.event == name]
        log['loo_' + name] = {'min': float(g.coef_without.min()), 'max': float(g.coef_without.max()),
                              'top5_by_abs_change': g.reindex(g.change.abs().sort_values(ascending=False).index)
                              .head(5)[['name', 'ai_pre', 'change']].round(4).to_dict('records')}

    # F event-month handling and window length
    for D, (name, shock, sign) in zip(parts, EVENTS):
        rec('F timing', 'event month dropped', name, single(D[D.k != -1]))
        Dp = D.copy()
        Dp.loc[Dp.k == -1, 'post'] = 1.0
        rec('F timing', 'event month coded post', name, single(Dp))
        rec('F timing', 'window +/-6 months', name, single(D[D.k.between(-6, 5)]))
        rec('F timing', 'event month and first post month dropped', name, single(D[~D.k.isin([-1, 0])]))
    rec('F timing', 'event month dropped', 'stacked', stacked(allD[allD.k != -1]))
    rec('F timing', 'window +/-6 months', 'stacked', stacked(allD[allD.k.between(-6, 5)]))
    rec('F timing', 'event month and first post month dropped', 'stacked', stacked(allD[~allD.k.isin([-1, 0])]))

    # G past returns and flow persistence
    for D, (name, shock, sign) in zip(parts, EVENTS):
        rec('G controls', '+ past 3/12-month returns', name, single(D, PR))
        rec('G controls', '+ flow lags 2-3', name, single(D, FL))
        rec('G controls', '+ both', name, single(D, PR + FL))
    rec('G controls', '+ past 3/12-month returns', 'stacked', stacked(allD, PR))
    rec('G controls', '+ flow lags 2-3', 'stacked', stacked(allD, FL))
    rec('G controls', '+ both', 'stacked', stacked(allD, PR + FL))

    # H industry composition
    IS = industry_shares()
    ind = ['ind_semis', 'ind_software', 'ind_hardware', 'ind_electronics', 'ind_commsvc']
    parts_i = []
    for D, (name, shock, sign) in zip(parts, EVENTS):
        P = pre_portfolio(EXP_PRE[name], shock).merge(IS, on=['series_id', 'q'], how='left')
        Di = D.merge(P[['series_id'] + ind], on='series_id', how='left')
        Di[ind] = Di[ind].fillna(0)
        extra = []
        for c in ind + ['tech_pre']:
            Di[c + '_post'] = Di[c] * Di.post
            extra.append(c + '_post')
        rec('H industry', '+ industry shares x post (and tech)', name, single(Di, extra))
        for c in ind + ['tech_pre']:
            Di[c + '_post_sign'] = Di[c + '_post'] * Di.sign
        parts_i.append(Di)
        log['corr_ai_pre_with_industry_' + name] = Di.groupby('series_id')[['ai_pre'] + ind + ['tech_pre']].first() \
            .corr().loc['ai_pre'].round(3).to_dict()
    DI = pd.concat(parts_i, ignore_index=True)
    extra = [c + s for c in ind + ['tech_pre'] for s in ('_post', '_post_sign')]
    rec('H industry', '+ industry shares x post x sign (and tech)', 'stacked', stacked(DI, extra))
    extra2 = [c + s for c in ['ind_semis', 'tech_pre'] for s in ('_post', '_post_sign')]
    rec('H industry', '+ semiconductors and tech only', 'stacked', stacked(DI, extra2))

    # I annual vintages
    parts_v = build_windows(L, {'chatgpt': EXP_MAIN, 'deepseek': EXP_MAIN})
    for D, (name, shock, sign) in zip(parts_v, EVENTS):
        rec('I annual vintages', 'baseline', name, single(D))
        rec('I annual vintages', '+ past returns', name, single(D, PR))
    allV = pd.concat(parts_v, ignore_index=True)
    rec('I annual vintages', 'baseline', 'stacked', stacked(allV))
    rec('I annual vintages', '+ past returns', 'stacked', stacked(allV, PR))
    # correlation between label sources at the fund level
    for (D1, D2, (name, _, _)) in zip(parts, parts_v, EVENTS):
        a = D1.groupby('series_id').ai_pre.first()
        b = D2.groupby('series_id').ai_pre.first()
        j = pd.concat([a, b], axis=1, join='inner')
        log['fund_exposure_corr_preevent_vs_vintage_' + name] = float(j.corr().iloc[0, 1])

    # J rolling dates with date-specific information
    roll = []
    for s in pd.date_range('2021-04-01', '2025-06-01', freq='MS'):
        S2.WIN = 12
        try:
            D = add_k(S2.window('roll', s, 1.0, EXP_MAIN, L), s)
            b, se, n, g = single(D)
            b2, se2, _, _ = single(D, PR)
            roll.append({'date': s.date(), 'coef': b, 'se': se, 't': b / se, 'n': n, 'funds': g,
                         'coef_past_returns': b2, 'se_past_returns': se2,
                         'actual_event': s in (EVENTS[0][1], EVENTS[1][1])})
        except Exception as exc:  # noqa: BLE001
            roll.append({'date': s.date(), 'error': str(exc)})
    R = pd.DataFrame(roll)
    R.to_csv(TABLES / 'event_date_scan.csv', index=False, float_format='%.6g')
    ok = R.dropna(subset=['coef'])
    for name, shock, sign in EVENTS:
        c = float(ok.loc[pd.to_datetime(ok.date) == shock, 'coef'].iloc[0])
        log['rolling_rank_' + name] = {
            'coef_at_actual_date_annual_vintage': c, 'dates': len(ok),
            'share_dates_coef_at_least_as_extreme_in_event_direction': float(np.mean(sign * ok.coef >= sign * c)),
            'share_dates_abs_t_above_2': float(np.mean(ok.t.abs() > 2)),
            'min_coef': float(ok.coef.min()), 'max_coef': float(ok.coef.max())}

    # K fund-level collapse
    krows = []
    for D, (name, shock, sign) in zip(parts, EVENTS):
        g = D.groupby(['series_id', D.post.astype(int)]).agg(nf=('nf_w', 'mean'), ret=('ret', 'mean'),
                                                              n=('nf_w', 'size')).unstack()
        g = g[(g[('n', 0)] >= 6) & (g[('n', 1)] >= 6)]
        f = D.groupby('series_id')[['ai_pre', 'claim_pre', 'lsize']].first().loc[g.index]
        y = g[('nf', 1)] - g[('nf', 0)]
        X = pd.DataFrame({'const': 1.0, 'ai_pre': f.ai_pre, 'claim_pre': f.claim_pre, 'lsize': f.lsize,
                          'dret': g[('ret', 1)] - g[('ret', 0)]})
        import statsmodels.api as sm
        m = sm.OLS(y.values, X.values).fit(cov_type='HC1')
        krows.append({'event': name, 'coef_ai_pre': m.params[1], 'se': m.bse[1], 'funds': len(y)})
        m2 = sm.OLS(y.values, X.drop(columns='dret').values).fit(cov_type='HC1')
        krows.append({'event': name + ' (no return change)', 'coef_ai_pre': m2.params[1], 'se': m2.bse[1], 'funds': len(y)})
    pd.DataFrame(krows).to_csv(TABLES / 'event_fund_collapse.csv', index=False, float_format='%.6g')
    log['fund_collapse'] = krows

    RES = pd.DataFrame(res)
    RES.to_csv(TABLES / 'event_extensions.csv', index=False, float_format='%.6g')
    (TABLES / 'event_extensions_log.json').write_text(json.dumps(log, indent=2, default=str), encoding='utf-8')
    pd.set_option('display.width', 250)
    print(RES.round(4).to_string())
    print(json.dumps(log, indent=2, default=str))


if __name__ == '__main__':
    main()

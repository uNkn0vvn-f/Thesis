"""The shift-level permutation test, on the two-event panel design itself.

shift_share/shift_share_two_events collapse each window to one cross-section of funds, the setting of Borusyak,
Hull and Jaravel. Collapsed, the direction response b is not distinguishable from what the
same AI labels would produce if reassigned across firms (studentised p about 0.1). But the
collapse is not the design the paper reports. The two-event panel keeps every
fund-month, absorbs fund-within-window and window x month effects, and controls for size,
the month's own return and the lagged flow; on it b is +0.138 (t 4.48), and - unlike the
collapse - it does not depend on NVIDIA (it rises to +0.157 without it). So the question
has to be asked of the panel too.

HOW, WITHOUT RE-ABSORBING THE FIXED EFFECTS 2,000 TIMES. A permutation changes only each
fund's pre-shock exposure, and the regressors are linear in it:

    ai_post_o      = ai_pre_{u(o)} * post_o           o = fund-month, u = fund-window
    ai_post_sign_o = sign_o * ai_post_o
    ai_pre_u       = sum_k wtilde_uk * g_k            (weights within the labelled sleeve)

so absorb(ai_post) = M @ ai_pre with M the absorbed matrix of post x fund-window
indicators, computed once; the window-constant sign passes through the within-window fixed
effects unchanged. Each permutation is then a matrix product and one clustered OLS.

The real labels must reproduce two_events' b and t exactly, or the script stops. One
permutation of firm identities is applied to both vintages, within technology x size
quintile (size on 29 November 2022, before both shocks), among firms labelled in both.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from src import paths as tp  # noqa: E402
import io
import os
import sys

import numpy as np
import pandas as pd

OUT = tp.out('analysis', 'shift_share', 'out')
LOGS = tp.out('analysis', 'shift_share', 'logs')
EXPS = {t: os.path.join(tp.out('analysis', 'preevent', '03_exposure', 'out'), t, 'fund_quarter_exposure.parquet')
        for t in ('chatgpt', 'deepseek')}
from src.analysis.events import two_events as T2                                            # noqa: E402
from src.analysis.events.event_dates import treat_cutoff                                   # noqa: E402
from src.analysis.flows.estimation import absorb, make_dims, absorbed_k               # noqa: E402
from src.analysis.flows.estimation import ols_cluster                               # noqa: E402
from src.analysis.events import shift_share as S1                                          # noqa: E402

N_PERM = int(os.environ.get('N_PERM', '2000'))
# PERM=techsize: within technology x size quintile (does the label matter beyond sector and
# size?); PERM=tech: within technology / other (does it matter beyond sector?)
PERM = os.environ.get('PERM', 'techsize')
SHOCK = {'chatgpt': pd.Timestamp('2022-12-01'), 'deepseek': pd.Timestamp('2025-02-01')}
SIGN = {'chatgpt': +1.0, 'deepseek': -1.0}


def sleeve_weights(tag, uids):
    """wtilde: each fund's pre-shock portfolio weights within its labelled sleeve, exactly
    the sleeve the exposure panel averages over; and that vintage's labels."""
    E = pd.read_parquet(EXPS[tag])
    E['report_date'] = pd.to_datetime(E.report_date)
    E['filing_date'] = pd.to_datetime(E.filing_date)
    pre = E[(E.filing_date < treat_cutoff(SHOCK[tag])) & (E.labelled >= 50)]
    # the latest row, not .last(): .last() fills each column from the last NON-NULL value,
    # so a missing field would silently come from an earlier quarter
    pre = pre.sort_values('report_date').groupby('series_id').tail(1).reset_index(drop=True)
    L = pd.read_parquet(os.path.join(tp.out('analysis', 'preevent', '02_label', 'out'), tag,
                                     'firm_ai_labels.parquet'), columns=['gvkey', 'ai_rev'])
    L['gvkey'] = L.gvkey.astype(str)
    # the exposure panel averages ai_rev only over firms where it exists (e01: ok = notna)
    L = L[L.ai_rev.notna()]
    H = pd.read_parquet(tp.out('holdings', 'holdings_bridged.parquet'),
                        columns=['series_id', 'report_date', 'gvkey', 'value'])
    H['report_date'] = pd.to_datetime(H.report_date)
    H = H[(H.value > 0) & H.gvkey.notna()].copy()
    H['gvkey'] = H.gvkey.astype(str)
    h = H.merge(pre[['series_id', 'report_date']], on=['series_id', 'report_date'])
    h = h.merge(L, on='gvkey', how='inner')
    h = h.groupby(['series_id', 'gvkey']).agg(value=('value', 'sum'),
                                              ai_rev=('ai_rev', 'first')).reset_index()
    h['w'] = h.value / h.groupby('series_id').value.transform('sum')
    h['uid'] = h.series_id + '|' + tag
    h = h[h.uid.isin(set(uids))]
    return h


def main():
    buf = io.StringIO()

    def w(s=''):
        buf.write(s + '\n')
        try:
            print(s)
        except UnicodeEncodeError:
            print(s.encode('ascii', 'replace').decode())

    # ---- the design, exactly as two_events builds it ----
    L = T2.panel()
    parts = [T2.window(t, SHOCK[t], SIGN[t], EXPS[t], L) for t in ('chatgpt', 'deepseek')]
    D = pd.concat(parts, ignore_index=True)
    D['cl_post'] = D.claim_pre * D.post
    D['cl_post_sign'] = D.claim_pre * D.post * D.sign
    D['ai_post'] = D.ai_pre * D.post
    D['ai_post_sign'] = D.ai_post * D.sign
    sub, dims = make_dims(D, [['uid'], ['win', 'month']])
    nabs = absorbed_k(dims, sub.series_id.values)
    Y = absorb(sub[['nf_w']].values, dims)[:, 0]
    Z = absorb(sub[['cl_post', 'cl_post_sign'] + T2.XC].values, dims)
    Xreal = absorb(sub[['ai_post', 'ai_post_sign']].values, dims)
    clus = sub.series_id.values

    def fit(xa, xs):
        X = np.column_stack([xa, xs, Z])
        b, se, *_ = ols_cluster(Y, X, clus, n_absorbed=nabs)
        return b[0], se[0], b[1], se[1]

    a0, sa0, b0, sb0 = fit(Xreal[:, 0], Xreal[:, 1])

    # ---- M: absorbed post x fund-window indicators, built once ----
    uids = sorted(sub.uid.unique())
    ui = {u: j for j, u in enumerate(uids)}
    col = sub.uid.map(ui).to_numpy()
    Q = np.zeros((len(sub), len(uids)))
    Q[np.arange(len(sub)), col] = sub.post.to_numpy()
    M = absorb(Q, dims)
    sgn = sub.sign.to_numpy()
    apre = sub.groupby('uid').ai_pre.first().reindex(uids).to_numpy()
    xa = M @ apre
    # the absorption is iterative, so column-by-column and combined absorption agree only
    # to its convergence tolerance: the check is relative, and b must survive it
    chk = np.abs(xa - Xreal[:, 0]).max() / np.abs(Xreal[:, 0]).max()
    assert chk < 1e-6, 'M @ ai_pre does not reproduce the absorbed regressor (%.2e)' % chk
    a1, _, b1, sb1 = fit(xa, sgn * xa)
    assert abs(b1 - b0) < 1e-6 * abs(b0) and abs(sb1 - sb0) < 1e-6 * sb0,         'reconstruction changed b: %.8f vs %.8f' % (b1, b0)

    # ---- sleeve weights and labels, and the reconstruction of ai_pre from them ----
    hw = pd.concat([sleeve_weights(t, uids) for t in ('chatgpt', 'deepseek')])
    firms = sorted(hw.gvkey.unique())
    fi = {k: j for j, k in enumerate(firms)}
    Wt = np.zeros((len(uids), len(firms)))
    Wt[hw.uid.map(ui).to_numpy(), hw.gvkey.map(fi).to_numpy()] = hw.w.to_numpy()
    g = {}
    for t in ('chatgpt', 'deepseek'):
        L_ = pd.read_parquet(os.path.join(tp.out('analysis', 'preevent', '02_label', 'out'), t,
                                          'firm_ai_labels.parquet'), columns=['gvkey', 'ai_rev'])
        L_['gvkey'] = L_.gvkey.astype(str)
        g[t] = L_.drop_duplicates('gvkey').set_index('gvkey').ai_rev.reindex(firms).to_numpy()
    is_t = {t: np.array([u.endswith('|' + t) for u in uids]) for t in g}

    def ai_pre_from(gc, gd):
        out = np.zeros(len(uids))
        out[is_t['chatgpt']] = Wt[is_t['chatgpt']] @ np.nan_to_num(gc)
        out[is_t['deepseek']] = Wt[is_t['deepseek']] @ np.nan_to_num(gd)
        return out

    rec = ai_pre_from(g['chatgpt'], g['deepseek'])
    err = np.abs(rec - apre).max()
    assert err < 1e-8, 'sleeve weights do not reproduce ai_pre (%.2e)' % err

    # ---- strata: technology x size quintile, among firms labelled in both vintages ----
    both = np.isfinite(g['chatgpt']) & np.isfinite(g['deepseek'])
    Uu = pd.read_parquet(tp.out('firm_labels', 'firm_universe.parquet'), columns=['gvkey', 'sector'])
    Uu['gvkey'] = Uu.gvkey.astype(str)
    tech = (Uu.drop_duplicates('gvkey').set_index('gvkey').sector.reindex(firms) == 'tech'
            ).astype(float).to_numpy()
    # Size is read on 29 November 2022, the last trading day before ChatGPT: dated before
    # both windows' shocks, so the strata carry no information from either.
    Dc, CL = S1.crsp_day(None)
    ev = pd.Timestamp('2022-11-29')
    dd = Dc[Dc.dlycaldt == ev][['permno', 'dlycap']].dropna()
    assert len(dd), 'no CRSP caps on the size date'
    dd = dd.merge(CL[['gvkey', 'permno', 'linkdt', 'linkenddt']], on='permno')
    dd = dd[(ev >= dd.linkdt) & (ev <= dd.linkenddt)]
    cap = dd.groupby('gvkey').dlycap.max().reindex(firms)
    qn = pd.qcut(cap.rank(method='first'), 5, labels=False).fillna(-1).to_numpy()
    key = np.where(both, tech * 10 + qn if PERM == 'techsize' else tech, -99)
    strata = [np.flatnonzero(key == v) for v in np.unique(key) if v != -99]

    rng = np.random.default_rng(20260924)
    P = np.empty((N_PERM, 4))
    for r in range(N_PERM):
        perm = np.arange(len(firms))
        for ix in strata:
            perm[ix] = ix[rng.permutation(len(ix))]
        ap = ai_pre_from(g['chatgpt'][perm], g['deepseek'][perm])
        x = M @ ap
        P[r] = fit(x, sgn * x)

    def pv(obs, dist):
        return (1 + np.sum(np.abs(dist) >= abs(obs))) / (1 + len(dist))

    ta, tb = a0 / sa0, b0 / sb0
    w('=' * 96)
    w('SHIFT-LEVEL PERMUTATION ON THE TWO-EVENT PANEL DESIGN (two shocks stacked)')
    w('=' * 96)
    w('  fund-months %d, fund-windows %d, firms %d (%d permuted, in %d strata)'
      % (len(sub), len(uids), len(firms), int(both.sum()), len(strata)))
    w('  reproduction: absorbed regressor max |diff| %.1e; ai_pre from sleeve weights %.1e'
      % (chk, err))
    w('  permutations: %d, one permutation of firm identities for both vintages, within %s'
      % (N_PERM, 'technology x size quintile' if PERM == 'techsize' else 'technology / other'))
    w()
    w('  %-30s %9s %8s %12s %12s' % ('', 'observed', 't', 'p (coef)', 'p (t)'))
    w('  %-30s %+9.4f %8.2f %12.4f %12.4f' % ('a, standing premium', a0, ta, pv(a0, P[:, 0]),
                                              pv(ta, P[:, 0] / P[:, 1])))
    w('  %-30s %+9.4f %8.2f %12.4f %12.4f' % ('b, response to direction', b0, tb, pv(b0, P[:, 2]),
                                              pv(tb, P[:, 2] / P[:, 3])))
    w()
    w('  permutation distribution of b: mean %+.4f, sd %.4f; of t(b): sd %.2f'
      % (P[:, 2].mean(), P[:, 2].std(), (P[:, 2] / P[:, 3]).std()))
    w('  A t-statistic whose permutation sd is well above 1 is the signature of fund-clustered')
    w('  errors understating the uncertainty that comes from which stocks carry the labels.')
    R = pd.DataFrame({'stat': ['a', 'b'], 'observed': [a0, b0], 't': [ta, tb],
                      'p_coef': [pv(a0, P[:, 0]), pv(b0, P[:, 2])],
                      'p_t': [pv(ta, P[:, 0] / P[:, 1]), pv(tb, P[:, 2] / P[:, 3])],
                      'perm_sd_coef': [P[:, 0].std(), P[:, 2].std()],
                      'perm_sd_t': [(P[:, 0] / P[:, 1]).std(), (P[:, 2] / P[:, 3]).std()]})
    R.to_parquet(os.path.join(OUT, 'panel_permutation_%s.parquet' % PERM), index=False)
    w()
    w('wrote out/panel_permutation_%s.parquet' % PERM)
    with io.open(os.path.join(LOGS, 'shift_share_panel_%s.txt' % PERM), 'w', encoding='utf-8') as fh:
        fh.write(buf.getvalue())


if __name__ == '__main__':
    main()

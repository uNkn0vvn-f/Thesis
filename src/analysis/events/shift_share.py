"""The shock designs read as shift-share designs, with the inference that reading requires.

WHY. A market-wide shock reaches a fund only through what it holds, so every treatment in
the two-event panel is already a shift-share variable: a fund's exposure is the sum over firms of its
pre-shock portfolio weight (the SHARE) times the firm's AI intensity (the SHIFT). Borusyak,
Hull and Jaravel (JEP 2025) set out what that implies. When identification comes from the
shifts, two funds holding the same AI stocks receive the same shock, so fund-clustered
standard errors - which treat them as independent - overstate precision; the correct ones
come from the equivalent regression at the level of the shifts (Adao, Kolesar and Morales
2019). The same framing asks for four things the designs did not report: a control for the
share of each portfolio the shifts cover (the "incomplete share" control), the effective
number of shifts, balance tests at the level of the shifts, and shares measured before the
natural experiment began.

WHAT THIS ESTIMATES, per shock, as one cross-section of funds:

    dflow_i = b * x_i + controls_i + e_i
    dflow_i   mean monthly net flow over months 0..11 minus months -12..-1 (winsorised as
              in two_events), for funds with at least 6 months on each side
    x_i       sum_k w_ik * AIrev_k : the portfolio-weighted AI sales share of the fund's
              last portfolio before the shock (pre-shock firm label from the pre-event information set);
              w_ik is the weight in ALL equity, not renormalised to the labelled sleeve
    controls  constant; the covered share sum_k w_ik over labelled firms (incomplete
              share); the technology share sum_k w_ik * tech_k (a shift-level control,
              share-aggregated, as in the two-shock design net of technology)

  A. SHIFTS = THE FIRM LABEL. Same design as two_events, in a cross-section. Reported with
     conventional HC1 errors and with exposure-robust errors from the shift-level
     regression, clustered by SIC2 industry because AI shifts are correlated within one.
  B. SHIFTS = THE MARKET'S OWN REPRICING (DeepSeek only). g_k is firm k's return on
     27 January 2025 minus the value-weighted CRSP market that day - the market's own
     measurement of how much each firm lost to the news, with no classifier in it.
     Reduced form: dflow on z_i = sum_k w_ik g_k. IV: x_i instrumented by z_i, so the
     label's measurement error is purged by a measure built from prices.
     Not done for ChatGPT: its repricing was spread over months (AI stocks fell in
     December 2022 and rallied from January), and any return window long enough to hold
     it also holds the March 2023 bank failures and Nvidia's May guidance.

Every exposure-robust estimate is checked to reproduce the unit-level coefficient exactly;
a mismatch stops the script.
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
os.makedirs(OUT, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)

from src.analysis.flows.estimation import MIN_TNA                       # noqa: E402

W = tp.data('wrds')
HOLD = tp.out('holdings', 'holdings_bridged.parquet')
LAD = tp.out('analysis', 'flow_ladder', 'out', 'fund_month_ladder.parquet')
UNI = tp.out('firm_labels', 'firm_universe.parquet')

TAG = os.environ.get('TAG', 'deepseek')
# `size`: the last trading day before the event, for the size strata
CFG_ALL = {'chatgpt': dict(shock='2022-12-01', event='2022-11-30', size='2022-11-29',
                           price=False),
           'deepseek': dict(shock='2025-02-01', event='2025-01-27', size='2025-01-24',
                            price=True)}
CFG = CFG_ALL[TAG]
SHOCK, EVENT = pd.Timestamp(CFG['shock']), pd.Timestamp(CFG['event'])
LAB = os.path.join(tp.out('analysis', 'preevent', '02_label', 'out'), TAG, 'firm_ai_labels.parquet')
EXP = os.path.join(tp.out('analysis', 'preevent', '03_exposure', 'out'), TAG, 'fund_quarter_exposure.parquet')
WIN, MIN_SIDE = 12, 6
# BAL=1 keeps only funds observed in all 24 months, so a fund launched shortly before the
# shock - whose early inflows decay whatever happens to AI - cannot drive post minus pre
BAL = os.environ.get('BAL', '0') == '1'
N_PERM = int(os.environ.get('N_PERM', '2000'))
# PERM=tech permutes AI labels among technology firms and among the rest; PERM=techsize
# also within market-cap quintile, so NVIDIA's label can only move to another mega-cap
PERM = os.environ.get('PERM', 'tech')


# --------------------------------------------------------------------------- estimators
def resid(y, W):
    b, *_ = np.linalg.lstsq(W, y, rcond=None)
    return y - W @ b


def unit_ols(y, x, W):
    """b on x with controls W, HC1 errors - the conventional answer."""
    X = np.column_stack([x, W])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    n, k = X.shape
    XtXi = np.linalg.pinv(X.T @ X)
    V = XtXi @ ((X * e[:, None]).T @ (X * e[:, None])) @ XtXi * n / (n - k)
    return b[0], np.sqrt(V[0, 0])


def unit_iv(y, x, z, W):
    """Just-identified IV with controls, HC1 errors."""
    zt, xt, yt = resid(z, W), resid(x, W), resid(y, W)
    b = (zt @ yt) / (zt @ xt)
    e = yt - b * xt
    n = len(y)
    se = np.sqrt(np.sum((zt * e) ** 2)) / abs(zt @ xt) * np.sqrt(n / (n - W.shape[1] - 1))
    return b, se


def shift_level(yt, xt, S, g, gctl, clus):
    """Exposure-robust inference from the equivalent shift-level regression.

    yt, xt : unit-level outcome and treatment, residualised on the unit controls
    S      : n_units x n_shifts share matrix
    g      : the shifts (instrument at the shift level)
    gctl   : shift-level controls matching the share-aggregated unit controls
    clus   : shift cluster ids (SIC2), since AI shifts are correlated within industry
    Returns b, clustered s.e., HC s.e., n_k, g residualised."""
    n_k = S.sum(axis=0)
    ok = n_k > 0
    S, g, gctl, clus, n_k = S[:, ok], g[ok], gctl[ok], clus[ok], n_k[ok]
    ybar = (S.T @ yt) / n_k
    xbar = (S.T @ xt) / n_k
    Wk = np.sqrt(n_k)[:, None]
    bg, *_ = np.linalg.lstsq(gctl * Wk, g * Wk[:, 0], rcond=None)
    gt = g - gctl @ bg
    den = np.sum(n_k * gt * xbar)
    b = np.sum(n_k * gt * ybar) / den
    u = n_k * gt * (ybar - b * xbar)
    se_hc = np.sqrt(np.sum(u ** 2)) / abs(den)
    cs = pd.Series(u).groupby(clus).sum().to_numpy()
    se_cl = np.sqrt(np.sum(cs ** 2)) / abs(den)
    # AKM0: the same variance with the null imposed (residual = outcome). The unrestricted
    # version above subtracts the fitted part first, and where the shift-level aggregates of
    # outcome and treatment line up closely - they are averages over overlapping sets of the
    # same funds - it collapses towards zero. Adao, Kolesar and Morales recommend AKM0 for
    # testing for exactly this reason.
    u0 = n_k * gt * ybar
    c0 = pd.Series(u0).groupby(clus).sum().to_numpy()
    shift_level.se0 = (np.sqrt(np.sum(u0 ** 2)) / abs(den), np.sqrt(np.sum(c0 ** 2)) / abs(den))
    return b, se_cl, se_hc, n_k, gt, ok


# --------------------------------------------------------------------------- data
def crsp_day(dates, need_mom_at=None):
    D = pd.read_parquet(os.path.join(W, 'crsp_dsf_v2_2020_2025.parquet'))
    D['dlycaldt'] = pd.to_datetime(D.dlycaldt)
    for c in ('dlyret', 'dlycap'):
        D[c] = pd.to_numeric(D[c], errors='coerce')
    CL = pd.read_parquet(os.path.join(W, 'ccm_linktable.parquet'))
    CL['gvkey'] = CL.gvkey.astype(str)
    CL = CL[CL.linktype.isin(['LU', 'LC']) & CL.linkprim.isin(['P', 'C'])]
    CL['linkdt'] = pd.to_datetime(CL.linkdt)
    CL['linkenddt'] = pd.to_datetime(CL.linkenddt).fillna(pd.Timestamp('2100-01-01'))
    return D, CL


def build(tag, bal):
    """One shock's cross-section: portfolios, shares, label shifts, outcome. Shared by
    shift_share (one shock at a time) and shift_share_two_events (both shocks under one permutation)."""
    cfg = CFG_ALL[tag]
    SHOCK = pd.Timestamp(cfg['shock'])
    EVENT = pd.Timestamp(cfg['event'])
    LAB = os.path.join(tp.out('analysis', 'preevent', '02_label', 'out'), tag, 'firm_ai_labels.parquet')
    EXP = os.path.join(tp.out('analysis', 'preevent', '03_exposure', 'out'), tag,
                       'fund_quarter_exposure.parquet')
    BAL = bal
    # ---- which portfolio stands for each fund: the two_events rule, exactly ----
    E = pd.read_parquet(EXP, columns=['series_id', 'report_date', 'filing_date', 'labelled'])
    E['report_date'] = pd.to_datetime(E.report_date)
    E['filing_date'] = pd.to_datetime(E.filing_date)
    # filed strictly before the event itself (event_dates.py)
    pre = E[(E.filing_date < EVENT) & (E.labelled >= 50)]
    # the latest row, not .last(): .last() fills each column from the last NON-NULL value,
    # so a missing field would silently come from an earlier quarter
    pre = pre.sort_values('report_date').groupby('series_id').tail(1).reset_index(drop=True)

    H = pd.read_parquet(HOLD, columns=['series_id', 'report_date', 'gvkey', 'value', 'sector',
                                       'conm'])
    H['report_date'] = pd.to_datetime(H.report_date)
    H = H.merge(pre[['series_id', 'report_date']], on=['series_id', 'report_date'])
    H = H[H.value > 0]
    tot = H.groupby('series_id').value.sum()
    H = H[H.gvkey.notna()].copy()
    H['gvkey'] = H.gvkey.astype(str)
    H = (H.groupby(['series_id', 'gvkey'])
         .agg(value=('value', 'sum'), sector=('sector', 'first'), conm=('conm', 'first'))
         .reset_index())
    H['w'] = H.value / H.series_id.map(tot)

    L = pd.read_parquet(LAB, columns=['gvkey', 'ai_rev'])
    L['gvkey'] = L.gvkey.astype(str)
    H = H.merge(L, on='gvkey', how='left')

    # ---- outcome: the two_events flow cleaning, collapsed to post minus pre ----
    F = pd.read_parquet(LAD, columns=['series_id', 'month', 'nf_beg', 'na_beg_recon', 'ret'])
    F['month'] = pd.to_datetime(F.month)
    F['nf'] = F.nf_beg.replace([np.inf, -np.inf], np.nan)
    F = F[F.na_beg_recon.notna() & (F.na_beg_recon >= MIN_TNA) & F.nf.notna() & F.ret.notna()]
    lo = F.groupby('month').nf.transform(lambda s: s.quantile(.01))
    hi = F.groupby('month').nf.transform(lambda s: s.quantile(.99))
    F['nf_w'] = F.nf.clip(lo, hi)
    F['k'] = (F.month.dt.year - SHOCK.year) * 12 + (F.month.dt.month - SHOCK.month)
    F = F[(F.k >= -WIN) & (F.k < WIN)]
    g = F.groupby('series_id')
    Y = pd.DataFrame({
        'post': F[F.k >= 0].groupby('series_id').nf_w.mean(),
        'pre': F[F.k < 0].groupby('series_id').nf_w.mean(),
        'n_post': F[F.k >= 0].groupby('series_id').size(),
        'n_pre': F[F.k < 0].groupby('series_id').size(),
        'pre_late': F[(F.k >= -6) & (F.k < 0)].groupby('series_id').nf_w.mean(),
        'pre_early': F[F.k < -6].groupby('series_id').nf_w.mean()})
    need = WIN if BAL else MIN_SIDE
    Y = Y[(Y.n_post >= need) & (Y.n_pre >= need)].copy()
    Y['dflow'] = Y.post - Y.pre
    Y['dflow_pre'] = Y.pre_late - Y.pre_early          # placebo: inside the pre-period

    funds = sorted(set(Y.index) & set(H.series_id))
    Y = Y.loc[funds]
    H = H[H.series_id.isin(funds)].copy()

    # ---- shift matrix: labelled firms ----
    lab = H[H.ai_rev.notna()]
    firms = sorted(lab.gvkey.unique())
    fi = {k: j for j, k in enumerate(firms)}
    ui = {s: i for i, s in enumerate(funds)}
    S = np.zeros((len(funds), len(firms)))
    S[lab.series_id.map(ui).to_numpy(), lab.gvkey.map(fi).to_numpy()] = lab.w.to_numpy()
    FK = lab.drop_duplicates('gvkey').set_index('gvkey').reindex(firms)
    g_lab = FK.ai_rev.to_numpy(dtype=float)
    tech_k = (FK.sector == 'tech').astype(float).to_numpy()
    U = pd.read_parquet(UNI, columns=['gvkey', 'sic4'])
    U['gvkey'] = U.gvkey.astype(str)
    sic2 = (U.drop_duplicates('gvkey').set_index('gvkey').sic4.reindex(firms)
            .astype(str).str.zfill(4).str[:2].fillna('xx').to_numpy())

    x = S @ g_lab
    cov = S.sum(axis=1)
    tech_all = (H.assign(t=(H.sector == 'tech') * H.w).groupby('series_id').t.sum()
                .reindex(funds).fillna(0).to_numpy())
    tech_lab = S @ tech_k
    y = Y.dflow.to_numpy(dtype=float)
    Wu = np.column_stack([np.ones(len(funds)), cov, tech_lab])
    return dict(pre=pre, funds=funds, firms=firms, S=S, g_lab=g_lab, tech_k=tech_k,
                sic2=sic2, x=x, cov=cov, tech_lab=tech_lab, y=y, Wu=Wu, Y=Y, FK=FK)


def main():
    buf = io.StringIO()

    def w(s=''):
        buf.write(s + '\n')
        try:
            print(s)
        except UnicodeEncodeError:
            print(s.encode('ascii', 'replace').decode())

    d = build(TAG, BAL)
    pre, funds, firms, S, g_lab, tech_k = (d[k] for k in ('pre', 'funds', 'firms', 'S',
                                                          'g_lab', 'tech_k'))
    sic2, x, cov, tech_lab, y, Wu, Y, FK = (d[k] for k in ('sic2', 'x', 'cov', 'tech_lab',
                                                         'y', 'Wu', 'Y', 'FK'))

    w('=' * 96)
    w('SHIFT-SHARE READING OF THE %s SHOCK  (post from %s, event %s)%s'
      % (TAG.upper(), SHOCK.date(), EVENT.date(),
         '   |   BALANCED: all 24 months' if BAL else ''))
    w('=' * 96)
    w('  funds %d | labelled firms (shifts) %d | pre-shock portfolios dated %s .. %s'
      % (len(funds), len(firms), pre.report_date.min().date(), pre.report_date.max().date()))
    w('  covered share of each portfolio (labelled firms): median %.1f%%, p10 %.1f%%'
      % (100 * np.median(cov), 100 * np.percentile(cov, 10)))
    w('  exposure x = sum w*AIrev : mean %.2f, sd %.2f (pp)' % (x.mean(), x.std()))
    w()

    # ---- shift descriptives: how many shocks really identify this ----
    n_k = S.sum(axis=0)
    sk = n_k / n_k.sum()
    w('-' * 96)
    w('1. THE SHIFTS')
    w('-' * 96)
    w('  effective number of shifts, 1 / sum(s_k^2) : %.1f  (of %d firms held)'
      % (1 / np.sum(sk ** 2), len(firms)))
    top = np.argsort(-sk)[:8]
    w('  largest exposure weights s_k:')
    for j in top:
        w('    %-40s s_k %5.2f%%   AI sales share %5.1f%%'
          % (str(FK.conm.iloc[j])[:40], 100 * sk[j], g_lab[j]))
    w('  weighted mean AI sales share across shifts %.1f%%, sd %.1f'
      % (np.average(g_lab, weights=n_k), np.sqrt(np.cov(g_lab, aweights=n_k))))
    w()

    # ---- A. label as shift ----
    yt, xt = resid(y, Wu), resid(x, Wu)
    b_u, se_u = unit_ols(y, x, Wu)
    gctl = np.column_stack([np.ones(len(firms)), tech_k])
    b_s, se_cl, se_hc, _, _, _ = shift_level(yt, xt, S, g_lab, gctl, sic2)
    # read the null-imposed errors NOW: shift_level keeps them on the function, and the
    # placebo call below overwrites them. Reading them after it reported the placebo's AKM0
    # error beside the main coefficient.
    s0hc, s0cl = shift_level.se0
    assert abs(b_s - b_u) < 1e-8 * max(1, abs(b_u)), 'shift-level b does not reproduce unit b'
    ypl = Y.dflow_pre.to_numpy(dtype=float)
    okp = np.isfinite(ypl)
    b_p, se_p_cl, _, _, _, _ = shift_level(resid(ypl[okp], Wu[okp]), resid(x[okp], Wu[okp]),
                                           S[okp], g_lab, gctl, sic2)
    bp_u, sep_u = unit_ols(ypl[okp], x[okp], Wu[okp])
    w('-' * 96)
    w('2. A. SHIFTS = FIRM AI LABEL   dflow on x, controls: covered share, technology share')
    w('-' * 96)
    w('  %-44s %9s %9s %7s' % ('', 'b', 'se', 't'))
    w('  %-44s %9.4f %9.4f %7.2f' % ('conventional (HC1, funds independent)', b_u, se_u, b_u / se_u))
    w('  %-44s %9.4f %9.4f %7.2f' % ('exposure-robust, shift-level HC', b_s, se_hc, b_s / se_hc))
    w('  %-44s %9.4f %9.4f %7.2f' % ('exposure-robust, shifts clustered by SIC2', b_s, se_cl,
                                      b_s / se_cl))
    w('  %-44s %9.4f %9.4f %7.2f' % ('AKM0 (null imposed), clustered by SIC2', b_s, s0cl,
                                      b_s / s0cl))
    w('  placebo inside the pre-period (months -6..-1 minus -12..-7):')
    w('  %-44s %9.4f %9.4f %7.2f' % ('  conventional', bp_u, sep_u, bp_u / sep_u))
    w('  %-44s %9.4f %9.4f %7.2f' % ('  exposure-robust, clustered', b_p, se_p_cl, b_p / se_p_cl))
    # Shift-level randomisation: the AI labels are permuted across firms, within the
    # technology / other strata the shift-level control defines, holding every portfolio
    # fixed. If the exposure-robust s.e. is right, it should match the spread of b under
    # permutation; if the design only works because of which funds hold what, the real b
    # will not stand out.
    rng = np.random.default_rng(20260924)
    if PERM == 'techsize':
        Dp, CLp = crsp_day(None)
        sz = pd.Timestamp(CFG['size'])
        dd = Dp[Dp.dlycaldt == sz][['permno', 'dlycap']].dropna()
        assert len(dd), 'no CRSP caps on the size date %s' % sz.date()
        dd = dd.merge(CLp[['gvkey', 'permno', 'linkdt', 'linkenddt']], on='permno')
        dd = dd[(sz >= dd.linkdt) & (sz <= dd.linkenddt)]
        cap = dd.groupby('gvkey').dlycap.max().reindex(firms)
        qn = pd.qcut(cap.rank(method='first'), 5, labels=False).fillna(-1).to_numpy()
        key = tech_k * 10 + qn
        strata = [np.flatnonzero(key == v) for v in np.unique(key)]
    else:
        strata = [np.flatnonzero(tech_k == v) for v in (0.0, 1.0)]

    def perm_b(yv, okv):
        out = np.empty(N_PERM)
        for r in range(N_PERM):
            gp = g_lab.copy()
            for ix in strata:
                gp[ix] = gp[rng.permutation(ix)]
            out[r] = unit_ols(yv[okv], (S @ gp)[okv], Wu[okv])[0]
        return out
    allok = np.ones(len(funds), bool)
    pb = perm_b(y, allok)
    ppl = perm_b(ypl, okp)
    p_main = (1 + np.sum(np.abs(pb) >= abs(b_u))) / (1 + N_PERM)
    p_pl = (1 + np.sum(np.abs(ppl) >= abs(bp_u))) / (1 + N_PERM)
    w('  shift-level randomisation (%d permutations of the labels across firms, within %s):'
      % (N_PERM, 'technology x size quintile' if PERM == 'techsize' else 'technology / other'))
    w('    main     b %+.4f   permutation sd %.4f   two-sided p %.4f' % (b_u, pb.std(), p_main))
    w('    placebo  b %+.4f   permutation sd %.4f   two-sided p %.4f' % (bp_u, ppl.std(), p_pl))
    rows = [dict(shock=TAG, design='A label shift', inference='conventional HC1', b=b_u, se=se_u),
            dict(shock=TAG, design='A label shift', inference='shift permutation sd', b=b_u,
                 se=pb.std()),
            dict(shock=TAG, design='A placebo pre-period', inference='shift permutation sd',
                 b=bp_u, se=ppl.std()),
            dict(shock=TAG, design='A label shift', inference='exposure-robust HC', b=b_s, se=se_hc),
            dict(shock=TAG, design='A label shift', inference='exposure-robust SIC2', b=b_s, se=se_cl),
            dict(shock=TAG, design='A label shift', inference='AKM0 SIC2', b=b_s, se=s0cl),
            dict(shock=TAG, design='A placebo pre-period', inference='exposure-robust SIC2',
                 b=b_p, se=se_p_cl)]

    # ---- firm characteristics at the event, for balance; and the event-day shift ----
    D, CL = crsp_day(None)
    day = D[D.dlycaldt == EVENT][['permno', 'dlyret', 'dlycap']].dropna()
    if True:
        mkt = np.average(day.dlyret, weights=day.dlycap)
        prior = D[(D.dlycaldt >= EVENT - pd.Timedelta(days=372))
                  & (D.dlycaldt < EVENT - pd.Timedelta(days=7))]
        mom = (prior.assign(gg=1 + prior.dlyret.fillna(0)).groupby('permno').gg.prod() - 1)
        day = day.merge(CL[['gvkey', 'permno', 'linkdt', 'linkenddt']], on='permno')
        day = day[(EVENT >= day.linkdt) & (EVENT <= day.linkenddt)]
        day = day.sort_values('dlycap').drop_duplicates('gvkey', keep='last')
        day['g'] = 100 * (day.dlyret - mkt)
        day['mom'] = day.permno.map(mom)
        day['lcap'] = np.log(day.dlycap)
        P = day.set_index('gvkey').reindex(firms)
        okk = P.g.notna().to_numpy()
        g_p = P.g.fillna(0).to_numpy()
        Sp = S * okk[None, :]
        z = Sp @ g_p
        cov_p = Sp.sum(axis=1)
        tech_p = Sp @ tech_k
        # the share-aggregated controls must be built on the same shifts the instrument uses,
        # or the shift-level regression is not the unit-level one
        Wp = np.column_stack([np.ones(len(funds)), cov, cov_p, tech_p])
        # balance at the shift level, on the firms with a price
        m = okk & np.isfinite(P.mom.to_numpy()) & np.isfinite(P.lcap.to_numpy())
        Xb = np.column_stack([np.ones(m.sum()), P.lcap.to_numpy()[m], P.mom.to_numpy()[m],
                              tech_k[m]])
        wb = np.sqrt(n_k[m])[:, None]
        w()
        w('  shift-level balance, weighted by s_k (firms with a price, n = %d):' % m.sum())
        cands = [('AI sales share', g_lab[m])]
        if CFG['price']:
            cands.append(('%s market-adjusted return' % EVENT.strftime('%d-%b'), g_p[m]))
        for nm, gv in cands:
            bb, *_ = np.linalg.lstsq(Xb * wb, gv * wb[:, 0], rcond=None)
            w('    %-34s on  log size %+7.3f   past-year return %+7.3f   tech %+7.3f'
              % (nm, bb[1], bb[2], bb[3]))
    if CFG['price']:
        r = np.corrcoef(g_lab[okk], g_p[okk])[0, 1]
        rw = np.cov(g_lab[okk], g_p[okk], aweights=n_k[okk])
        w('  across firms, AI label vs 27-Jan return: r = %.3f unweighted, %.3f weighted by s_k'
          % (r, rw[0, 1] / np.sqrt(rw[0, 0] * rw[1, 1])))
        w('  firms with a price on the event day: %d of %d, %.1f%% of the exposure weight'
          % (okk.sum(), len(firms), 100 * n_k[okk].sum() / n_k.sum()))
        w()
        w('-' * 96)
        w('3. B. SHIFTS = THE MARKET\'S OWN REPRICING ON %s (no classifier)' % EVENT.date())
        w('-' * 96)
        w('  market that day (value-weighted CRSP): %+.2f%%' % (100 * mkt))
        w('  exposure z = sum w*g : mean %.3f pp, sd %.3f' % (z.mean(), z.std()))
        yt2, zt2 = resid(y, Wp), resid(z, Wp)
        b_r, se_r = unit_ols(y, z, Wp)
        gctl_p = np.column_stack([np.ones(len(firms)), tech_k])
        b_rs, se_rcl, se_rhc, _, _, _ = shift_level(yt2, zt2, Sp, g_p, gctl_p, sic2)
        assert abs(b_rs - b_r) < 1e-8 * max(1, abs(b_r)), 'reduced form does not reproduce'
        w('  REDUCED FORM  dflow on z (per pp of event-day portfolio loss relative to market)')
        w('  %-44s %9.4f %9.4f %7.2f' % ('conventional', b_r, se_r, b_r / se_r))
        w('  %-44s %9.4f %9.4f %7.2f' % ('exposure-robust, clustered', b_rs, se_rcl, b_rs / se_rcl))
        # first stage and IV: label exposure instrumented by price exposure
        b_f, se_f = unit_ols(x, z, Wp)
        xt2 = resid(x, Wp)
        b_fs, se_fcl, _, _, _, _ = shift_level(xt2, zt2, Sp, g_p, gctl_p, sic2)
        b_iv, se_iv = unit_iv(y, x, z, Wp)
        b_ivs, se_ivcl, se_ivhc, _, _, _ = shift_level(yt2, xt2, Sp, g_p, gctl_p, sic2)
        assert abs(b_ivs - b_iv) < 1e-8 * max(1, abs(b_iv)), 'IV does not reproduce'
        w('  FIRST STAGE   x (label exposure) on z')
        w('  %-44s %9.4f %9.4f %7.2f   F = %.1f' % ('conventional', b_f, se_f, b_f / se_f,
                                                   (b_f / se_f) ** 2))
        w('  %-44s %9.4f %9.4f %7.2f   F = %.1f' % ('exposure-robust, clustered', b_fs, se_fcl,
                                                   b_fs / se_fcl, (b_fs / se_fcl) ** 2))
        w('  IV            dflow on x, instrumented by z (per pp of AI sales exposure)')
        w('  %-44s %9.4f %9.4f %7.2f' % ('conventional', b_iv, se_iv, b_iv / se_iv))
        w('  %-44s %9.4f %9.4f %7.2f' % ('exposure-robust, clustered', b_ivs, se_ivcl,
                                          b_ivs / se_ivcl))
        w('  for comparison, OLS on x with the same controls: %.4f' % unit_ols(y, x, Wp)[0])
        rows += [dict(shock=TAG, design='B price shift, reduced form', inference='conventional HC1',
                      b=b_r, se=se_r),
                 dict(shock=TAG, design='B price shift, reduced form',
                      inference='exposure-robust SIC2', b=b_rs, se=se_rcl),
                 dict(shock=TAG, design='B first stage', inference='exposure-robust SIC2',
                      b=b_fs, se=se_fcl),
                 dict(shock=TAG, design='B IV, label instrumented by price',
                      inference='conventional HC1', b=b_iv, se=se_iv),
                 dict(shock=TAG, design='B IV, label instrumented by price',
                      inference='exposure-robust SIC2', b=b_ivs, se=se_ivcl)]
    else:
        w('    (price shifts not built for this shock; see the module docstring)')

    R = pd.DataFrame(rows)
    R['t'] = R.b / R.se
    sfx = TAG + ('_balanced' if BAL else '') + ('_techsize' if PERM == 'techsize' else '')
    R.to_parquet(os.path.join(OUT, 'shift_share_%s.parquet' % sfx), index=False)
    w()
    w('wrote out/shift_share_%s.parquet' % sfx)
    with io.open(os.path.join(LOGS, 'shift_share_%s.txt' % sfx), 'w', encoding='utf-8') as fh:
        fh.write(buf.getvalue())


if __name__ == '__main__':
    main()

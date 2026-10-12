"""Recentred shift-share estimates: what the AI label does beyond the composition of holdings.

WHY. shift_share_panel permutes firm labels within technology x size strata and finds the observed
direction response b = +0.138 unremarkable on the raw coefficient (p 0.35): the permuted
b averages +0.116, not zero. That is not noise. A fund's exposure under a permutation is
still its portfolio weights times labels drawn from the same strata, so funds that hold
large technology firms receive high exposure under ANY permutation. Exposure is non-random
even when the labels are, and the permutation null is centred on what the strata alone
predict.

Borusyak and Hull (2023, Econometrica, "Non-random exposure to exogenous shocks") give the
remedy. Compute each fund's EXPECTED exposure under the assignment process,

    mu_u = E[ ai_pre_u | permutation within strata ] = sum_k wtilde_uk * E[g_pi(k)],

which for a uniform permutation within a stratum is the stratum mean of the label, and
use the RECENTRED exposure ai_pre_u - mu_u. The recentred regression is consistent for the
effect of the labels themselves under the design assumption (labels as good as randomly
assigned within strata), and randomisation inference on it is exact. Here mu is analytic;
the script checks it against the mean of simulated permutations.

WHAT IS REPORTED, for the two-event panel (the paper's design) and for the collapsed
cross-section of shift_share/shift_share_two_events, and for five assignment processes (strata):

    none      labels permuted across all firms labelled in both vintages
    size      within market-cap quintile (29 November 2022, before both shocks)
    tech      within technology / other
    techsize  within technology x size quintile            (shift_share_panel's strata)
    sic2      within two-digit SIC industry

    raw       b on ai_pre, the two_events estimate
    mu only   b on the expected exposure alone: what the strata predict
    recentred b on ai_pre - mu                 (Borusyak-Hull)
    rec+mu    b on ai_pre - mu, with mu's own terms as controls

with fund-clustered t, and randomisation p-values from 2,000 permutations of firm
identities (one permutation applied to both vintages), raw and studentised. The panel's
unrecentred techsize permutation must reproduce shift_share_panel's p-values exactly, or the script stops.
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
from src.analysis.events import shift_share_panel as S3                                    # noqa: E402
from src.analysis.events import shift_share as S1                                          # noqa: E402
T2, absorb, make_dims, absorbed_k, ols_cluster = (S3.T2, S3.absorb, S3.make_dims,
                                                  S3.absorbed_k, S3.ols_cluster)

N_PERM = int(os.environ.get('N_PERM', '2000'))
SEED = 20260924
STRATA = ('none', 'size', 'tech', 'techsize', 'sic2')
SHOCK, SIGN = S3.SHOCK, S3.SIGN


_CAP = {}


def event_caps():
    """market cap by gvkey on 29 November 2022, the last trading day before ChatGPT, through
    the CCM link valid that day (read once). Dated before both shocks."""
    if 'cap' not in _CAP:
        Dc, CL = S1.crsp_day(None)
        ev = pd.Timestamp('2022-11-29')
        dd = Dc[Dc.dlycaldt == ev][['permno', 'dlycap']].dropna()
        assert len(dd), 'no CRSP caps on the size date'
        dd = dd.merge(CL[['gvkey', 'permno', 'linkdt', 'linkenddt']], on='permno')
        dd = dd[(ev >= dd.linkdt) & (ev <= dd.linkenddt)]
        _CAP['cap'] = dd.groupby('gvkey').dlycap.max()
    return _CAP['cap']


def firm_attributes(firms, tech=None):
    """technology flag, size quintile on 29 November 2022 (shift_share_panel's exact construction), SIC2.
    `tech` overrides the universe's flag, for the collapse, which shift_share_two_events builds from the
    holdings file's sector."""
    Uu = pd.read_parquet(tp.out('firm_labels', 'firm_universe.parquet'), columns=['gvkey', 'sector', 'sic4'])
    Uu['gvkey'] = Uu.gvkey.astype(str)
    Uu = Uu.drop_duplicates('gvkey').set_index('gvkey').reindex(firms)
    if tech is None:
        tech = (Uu.sector == 'tech').astype(float).to_numpy()
    sic2 = Uu.sic4.astype(str).str.zfill(4).str[:2].where(Uu.sic4.notna(), 'xx').to_numpy()
    cap = event_caps().reindex(firms)
    qn = pd.qcut(cap.rank(method='first'), 5, labels=False).fillna(-1).to_numpy()
    return tech, qn, sic2


def strata_of(kind, both, tech, qn, sic2):
    """Index arrays of the firms permuted together; firms outside `both` are never moved."""
    if kind == 'none':
        key = np.zeros(len(both))
    elif kind == 'size':
        key = qn.astype(float)
    elif kind == 'tech':
        key = tech
    elif kind == 'techsize':
        key = tech * 10 + qn
    elif kind == 'sic2':
        codes = {s: i for i, s in enumerate(sorted(set(sic2)))}
        key = np.array([codes[s] for s in sic2], dtype=float)
    key = np.where(both, key, -99)
    return [np.flatnonzero(key == v) for v in np.unique(key) if v != -99]


def expected_labels(g, strata):
    """E[g_pi(k)] under a uniform permutation within each stratum: the stratum mean for
    permuted firms, the firm's own label for the rest."""
    e = g.copy()
    for ix in strata:
        e[ix] = np.mean(g[ix])
    return e


def draw(rng, n, strata):
    perm = np.arange(n)
    for ix in strata:
        perm[ix] = ix[rng.permutation(len(ix))]
    return perm


def pv(obs, dist):
    return (1 + np.sum(np.abs(dist) >= abs(obs))) / (1 + len(dist))


# --------------------------------------------------------------------------- the panel
def panel():
    """The two-event panel, exactly as shift_share_panel builds it, with the pieces a permutation needs."""
    L = T2.panel()
    parts = [T2.window(t, SHOCK[t], SIGN[t], S3.EXPS[t], L) for t in ('chatgpt', 'deepseek')]
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
    uids = sorted(sub.uid.unique())
    ui = {u: j for j, u in enumerate(uids)}
    Q = np.zeros((len(sub), len(uids)))
    Q[np.arange(len(sub)), sub.uid.map(ui).to_numpy()] = sub.post.to_numpy()
    M = absorb(Q, dims)
    sgn = sub.sign.to_numpy()
    apre = sub.groupby('uid').ai_pre.first().reindex(uids).to_numpy()
    hw = pd.concat([S3.sleeve_weights(t, uids) for t in ('chatgpt', 'deepseek')])
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

    assert np.abs(ai_pre_from(g['chatgpt'], g['deepseek']) - apre).max() < 1e-8
    assert np.abs(M @ apre - Xreal[:, 0]).max() / np.abs(Xreal[:, 0]).max() < 1e-6
    both = np.isfinite(g['chatgpt']) & np.isfinite(g['deepseek'])
    return dict(Y=Y, Z=Z, M=M, sgn=sgn, clus=clus, nabs=nabs, apre=apre, g=g, firms=firms,
                both=both, ai_pre_from=ai_pre_from, n_fm=len(sub), n_u=len(uids),
                n_funds=sub.series_id.nunique())


def panel_fit(P, x, ctl=None):
    """a, se(a), b, se(b) for fund-window exposure x (post and post x sign), the two_events
    controls, and optionally a second exposure's two terms as controls."""
    xa = P['M'] @ x
    cols = [xa, P['sgn'] * xa, P['Z']]
    if ctl is not None:
        ca = P['M'] @ ctl
        cols += [ca[:, None], (P['sgn'] * ca)[:, None]]
    X = np.column_stack(cols)
    b, se, *_ = ols_cluster(P['Y'], X, P['clus'], n_absorbed=P['nabs'])
    return b[0], se[0], b[1], se[1]


def run_panel(P, attrs, w, rows):
    g1, g2 = P['g']['chatgpt'], P['g']['deepseek']
    n = len(P['firms'])
    a_raw, sa_raw, b_raw, sb_raw = panel_fit(P, P['apre'])
    w('PANEL (two-event design): %d fund-months, %d fund-windows, %d funds, %d firms (%d permuted)'
      % (P['n_fm'], P['n_u'], P['n_funds'], n, int(P['both'].sum())))
    w('  raw b %+.4f (t %.2f), a %+.4f (t %.2f): must equal two_events'
      % (b_raw, b_raw / sb_raw, a_raw, a_raw / sa_raw))
    w()
    w('  %-9s %-10s %9s %7s %11s %9s %9s %9s' % ('strata', 'estimate', 'b', 't', 'perm mean',
                                              'perm sd', 'p(coef)', 'p(t)'))
    for kind in STRATA:
        st = strata_of(kind, P['both'], *attrs)
        mu = P['ai_pre_from'](expected_labels(g1, st), expected_labels(g2, st))
        # mu checked against simulation: for every fund-window, the mean of 2,000 permuted
        # exposures minus mu, in units of its own simulation standard error. The check is
        # on the whole distribution of z, not its maximum: a concentrated fund's permuted
        # exposure is skewed (most labels are zero), so one fund-window in a thousand can
        # sit at |z| 4 to 7 by chance, while a wrong mu would shift every z and grow with
        # the number of draws. Standard normal: mean |z| 0.80, P(|z| > 4) 0.00006.
        rng = np.random.default_rng(SEED + 1)
        sims = np.array([P['ai_pre_from'](g1[p], g2[p])
                         for p in (draw(rng, n, st) for _ in range(2000))])
        sd = sims.std(axis=0, ddof=1)
        ok = sd > 1e-12
        z = (sims.mean(axis=0)[ok] - mu[ok]) / (sd[ok] / np.sqrt(len(sims)))
        mu_err = np.abs(z).mean()
        assert 0.6 < mu_err < 1.0 and (np.abs(z) > 4).mean() < 0.01, \
            'analytic mu disagrees with simulation (mean |z| %.2f, %.1f%% beyond 4)' % (
                mu_err, 100 * (np.abs(z) > 4).mean())
        if (~ok).any():                    # funds holding no permuted firm: mu is exact
            assert np.abs(sims.mean(axis=0)[~ok] - mu[~ok]).max() < 1e-9
        est = {'raw': (P['apre'], None), 'mu only': (mu, None),
               'recentred': (P['apre'] - mu, None), 'rec+mu': (P['apre'] - mu, mu)}
        obs = {k: panel_fit(P, x, c) for k, (x, c) in est.items()}
        rng = np.random.default_rng(SEED)
        D = {k: np.empty((N_PERM, 2)) for k in ('raw', 'recentred', 'rec+mu')}
        for r in range(N_PERM):
            p = draw(rng, n, st)
            ap = P['ai_pre_from'](g1[p], g2[p])
            for k, (x, c) in (('raw', (ap, None)), ('recentred', (ap - mu, None)),
                              ('rec+mu', (ap - mu, mu))):
                _, _, bb, sb = panel_fit(P, x, c)
                D[k][r] = (bb, bb / sb)
        if kind == 'techsize' and N_PERM == 2000:
            ref = pd.read_parquet(os.path.join(OUT, 'panel_permutation_techsize.parquet'))
            ref = ref[ref.stat == 'b'].iloc[0]
            got = (pv(b_raw, D['raw'][:, 0]), pv(b_raw / sb_raw, D['raw'][:, 1]))
            assert abs(got[0] - ref.p_coef) < 1e-12 and abs(got[1] - ref.p_t) < 1e-12, \
                'techsize raw permutation does not reproduce shift_share_panel: %s vs %s' % (
                    got, (ref.p_coef, ref.p_t))
        for k in ('raw', 'mu only', 'recentred', 'rec+mu'):
            _, _, bb, sb = obs[k]
            if k in D:
                dist = D[k]
                pm, psd = dist[:, 0].mean(), dist[:, 0].std()
                pc, pt = pv(bb, dist[:, 0]), pv(bb / sb, dist[:, 1])
            else:
                pm = psd = pc = pt = np.nan
            w('  %-9s %-10s %+9.4f %7.2f %+11.4f %9.4f %9.4f %9.4f'
              % (kind, k, bb, bb / sb, pm, psd, pc, pt))
            a_, sa_ = obs[k][0], obs[k][1]
            rows.append(dict(design='panel', strata=kind, estimate=k, b=bb, se=sb, t=bb / sb,
                             a=a_, se_a=sa_, perm_mean=pm, perm_sd=psd, p_coef=pc, p_t=pt,
                             n_strata=len(st), mu_sim_err=mu_err,
                             sd_mu=np.std(mu), sd_rec=np.std(P['apre'] - mu)))
        w('  %-9s %d strata; mu vs 2,000 simulated draws: mean |z| %.2f; sd of exposure: '
          'raw %.2f, mu %.2f, recentred %.2f'
          % ('', len(st), mu_err, np.std(P['apre']), np.std(mu), np.std(P['apre'] - mu)))
        w()


# --------------------------------------------------------------------------- the collapse
def run_collapsed(bal, w, rows):
    d1, d2 = S1.build('chatgpt', bal), S1.build('deepseek', bal)
    common = sorted(set(d1['firms']) & set(d2['firms']))
    p1 = {k: j for j, k in enumerate(d1['firms'])}
    p2 = {k: j for j, k in enumerate(d2['firms'])}
    i1 = np.array([p1[k] for k in common])
    i2 = np.array([p2[k] for k in common])
    tech, qn, sic2 = firm_attributes(common, tech=d2['tech_k'][i2])
    both = np.ones(len(common), bool)

    def slopes(x1, x2):
        b1, s1 = S1.unit_ols(d1['y'], x1, d1['Wu'])
        b2, s2 = S1.unit_ols(d2['y'], x2, d2['Wu'])
        return b1, b2, s1, s2

    def exposures(gc, gd):
        """fund exposures in both windows when the common firms carry labels gc, gd"""
        q1, q2 = d1['g_lab'].copy(), d2['g_lab'].copy()
        q1[i1], q2[i2] = gc, gd
        return d1['S'] @ q1, d2['S'] @ q2

    gc, gd = d1['g_lab'][i1], d2['g_lab'][i2]
    x1, x2 = exposures(gc, gd)
    w('COLLAPSED CROSS-SECTION (shift_share/shift_share_two_events design, %s): %d + %d funds, %d common firms permuted'
      % ('balanced' if bal else 'all funds', len(d1['funds']), len(d2['funds']), len(common)))
    w('  %-9s %-10s %9s %7s %11s %9s %9s %9s' % ('strata', 'estimate', 'b', 't', 'perm mean',
                                              'perm sd', 'p(coef)', 'p(t)'))
    for kind in STRATA:
        st = strata_of(kind, both, tech, qn, sic2)
        m1, m2 = exposures(expected_labels(gc, st), expected_labels(gd, st))

        def stat(e1, e2):
            b1, b2, s1, s2 = slopes(e1, e2)
            return (b1 - b2) / 2, (b1 - b2) / np.sqrt(s1 ** 2 + s2 ** 2)

        obs = {'raw': stat(x1, x2), 'mu only': stat(m1, m2), 'recentred': stat(x1 - m1, x2 - m2)}
        rng = np.random.default_rng(SEED)
        D = {k: np.empty((N_PERM, 2)) for k in ('raw', 'recentred')}
        for r in range(N_PERM):
            p = draw(rng, len(common), st)
            e1, e2 = exposures(gc[p], gd[p])
            D['raw'][r] = stat(e1, e2)
            D['recentred'][r] = stat(e1 - m1, e2 - m2)
        if bal and kind in ('tech', 'techsize') and N_PERM == 2000:
            ref = pd.read_parquet(os.path.join(OUT, 'two_shock_permutation_balanced_%s.parquet'
                                               % kind)).set_index('stat')
            got = (pv(obs['raw'][0], D['raw'][:, 0]), pv(obs['raw'][1], D['raw'][:, 1]))
            want = (ref.loc['b', 'p'], ref.loc['t b', 'p'])
            assert max(abs(got[0] - want[0]), abs(got[1] - want[1])) < 1e-12, \
                'collapsed %s raw permutation does not reproduce shift_share_two_events: %s vs %s' % (kind, got, want)
        for k in ('raw', 'mu only', 'recentred'):
            bb, tt = obs[k]
            if k in D:
                pm, psd = D[k][:, 0].mean(), D[k][:, 0].std()
                pc, pt = pv(bb, D[k][:, 0]), pv(tt, D[k][:, 1])
            else:
                pm = psd = pc = pt = np.nan
            w('  %-9s %-10s %+9.4f %7.2f %+11.4f %9.4f %9.4f %9.4f'
              % (kind, k, bb, tt, pm, psd, pc, pt))
            rows.append(dict(design='collapsed_' + ('balanced' if bal else 'all'), strata=kind,
                             estimate=k, b=bb, se=bb / tt if tt else np.nan, t=tt, a=np.nan,
                             se_a=np.nan, perm_mean=pm, perm_sd=psd, p_coef=pc, p_t=pt,
                             n_strata=len(st), mu_sim_err=np.nan, sd_mu=np.nan, sd_rec=np.nan))
        w()


def main():
    buf = io.StringIO()

    def w(s=''):
        buf.write(s + '\n')
        try:
            print(s)
        except UnicodeEncodeError:
            print(s.encode('ascii', 'replace').decode())

    w('=' * 100)
    w('RECENTRED SHIFT-SHARE ESTIMATES (Borusyak and Hull 2023): THE LABEL BEYOND COMPOSITION')
    w('=' * 100)
    w('  %d permutations of firm identities per assignment process, one permutation for both'
      % N_PERM)
    w('  vintages; p two-sided. "perm mean" is where the null distribution is centred: for the')
    w('  raw estimate it is what the strata alone predict, for the recentred one it should be ~0.')
    w()
    rows = []
    P = panel()
    run_panel(P, firm_attributes(P['firms']), w, rows)
    for bal in (True, False):
        run_collapsed(bal, w, rows)
    R = pd.DataFrame(rows)
    R.to_parquet(os.path.join(OUT, 'recentred.parquet'), index=False)
    w('wrote out/recentred.parquet')
    with io.open(os.path.join(LOGS, 'shift_share_recentred.txt'), 'w', encoding='utf-8') as fh:
        fh.write(buf.getvalue())


if __name__ == '__main__':
    main()

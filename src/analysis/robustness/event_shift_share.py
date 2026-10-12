"""Shift-share reading of the event design on the selected labels.

A market-wide shock reaches a fund through its holdings, so the event design's treatment is a
shift-share variable: pre-event portfolio weights (the shares) times company AI-revenue
labels (the shifts). Borusyak, Hull and Jaravel (2025) ask for inference at the level of the
shifts, a control for the share each portfolio's labelled firms cover, and a statement of
what makes the shifts as good as random; Borusyak and Hull (2023) recentre the exposure on
its expectation under the assignment process. The modules in src/analysis/events do this;
this script runs them on the pre-event labels that THESIS_LABELS selects (common.py) and
collects their outputs as CSV tables in results/tables/robustness/[deepseek/].

    single  one shock at a time, collapsed to post minus pre: conventional, exposure-robust
          (AKM, AKM0) errors, shift-level permutation of the labels, balance; for DeepSeek
          the label-free price shift as reduced form and instrument
    two_events  the sign reversal under one permutation of firm labels across both shocks
    panel  the same permutation on the stacked panel design
    leave_out  the panel without the largest AI stocks, and net of the technology share
    recentred  recentred estimates, panel and collapsed, five assignment processes (asserts that
          its unrecentred permutations reproduce two_events and panel)
    placebo  the pre-period placebo on the panel
    extra  (this script) permutations within technology x size x holding-weight strata, the
          per-standard-deviation statistic, and the covered-share x post controls in the panel

The event modules resolve the pre-event labels through src.paths.out('analysis',
'preevent', ...) and write to out('analysis', 'shift_share', ...); on the Luna labels both
are redirected to the *_luna folders before the scripts are imported. They read the
technology sector from the holdings file, whereas the exposure builder
(src.holdings.ai_exposure) takes it from the firm universe, and the two differ for 292 of
the 2,917 universe firms held; the scripts are therefore given a copy of the holdings file
with the universe's sector (data/interim/analysis/shift_share/inputs/), so that their
reconstruction of each fund's exposure and technology share reproduces the exposure panel
(panel, leave_out and recentred check this). Needs data/raw/wrds/crsp_dsf_v2_2020_2025.parquet (size
strata; DeepSeek-day returns).

    python src/analysis/robustness/event_shift_share.py [single two_events ...]   selected stages
    N_PERM=50 python ...                                                   quick check
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src import paths as tp  # noqa: E402
from src.analysis.robustness.common import LABELS, TABLES  # noqa: E402

_out = tp.out
HOLD_SRC = Path(_out('holdings', 'holdings_bridged.parquet'))
HOLD = Path(_out('analysis', 'shift_share', 'inputs', 'holdings_universe_sector.parquet'))


def out(name, *parts):
    """paths.out with the pre-event labels and the shift-share outputs of the Luna labels,
    and the holdings file carrying the universe's sector."""
    if name == 'holdings' and parts == ('holdings_bridged.parquet',):
        return str(HOLD)
    if (name == 'analysis' and parts and LABELS == 'luna'
            and parts[0] in ('preevent', 'shift_share')):
        return _out('analysis', parts[0] + '_luna', *parts[1:])
    return _out(name, *parts)


# The event modules resolve their paths through tp.out when imported; route them to the
# folders of the selected label set before the imports below.
tp.out = out


def holdings_with_universe_sector():
    """holdings_bridged with `sector` from the firm universe, as the exposure builder
    defines the technology share; outside the universe the sector is missing, which the
    builder also counts as not technology. Written on every run from the current file."""
    H = pd.read_parquet(HOLD_SRC)
    U = pd.read_parquet(_out('firm_labels', 'firm_universe.parquet'), columns=['gvkey', 'sector'])
    U['gvkey'] = U.gvkey.astype(str)
    U = U.drop_duplicates('gvkey').set_index('gvkey').sector
    H['sector'] = H.gvkey.astype(str).map(U)
    HOLD.parent.mkdir(parents=True, exist_ok=True)
    H.to_parquet(HOLD, index=False)
    return H.sector.notna().sum()
N_PERM = int(os.environ.get('N_PERM', '2000'))
os.environ['N_PERM'] = str(N_PERM)
from src.analysis.events import shift_share as S1  # noqa: E402
from src.analysis.events import shift_share_two_events as S2  # noqa: E402
from src.analysis.events import shift_share_panel as S3  # noqa: E402
from src.analysis.events import shift_share_leave_out as S4  # noqa: E402
from src.analysis.events import shift_share_recentred as S5  # noqa: E402
from src.analysis.events import shift_share_placebo as S6  # noqa: E402

OUT = Path(S1.OUT)
assert str(OUT).endswith(os.path.join('shift_share' + ('_luna' if LABELS == 'luna' else ''), 'out'))
# Check that each module reads the pre-event exposures of the selected label set.
for mod in (S3, S4, S6):
    for tag in ('chatgpt', 'deepseek'):
        want = str(Path(_out('analysis', 'preevent' + ('_luna' if LABELS == 'luna' else ''),
                             '03_exposure', 'out', tag, 'fund_quarter_exposure.parquet')))
        got = (mod.EXPS if hasattr(mod, 'EXPS') else mod.X)[tag]
        assert os.path.normcase(got) == os.path.normcase(want), (mod.__name__, got)
SAMPLES = ((False, 'tech'), (True, 'tech'), (True, 'techsize'))


def sfx(tag, bal, perm):
    """File-name suffix of one sample: event, balanced panel, technology x size strata."""
    return tag + ('_balanced' if bal else '') + ('_techsize' if perm == 'techsize' else '')


def shifts(tag, bal):
    """The shift descriptives of single's log: funds, firms, effective number of shifts,
    covered share, exposure moments, the largest exposure weight."""
    d = S1.build(tag, bal)
    n_k = d['S'].sum(axis=0)
    sk = n_k / n_k.sum()
    j = int(np.argmax(sk))
    return dict(shock=tag, sample='balanced' if bal else 'all funds', funds=len(d['funds']),
                firms=len(d['firms']), effective_shifts=1 / np.sum(sk ** 2),
                covered_share_median=100 * np.median(d['cov']),
                covered_share_p10=100 * np.percentile(d['cov'], 10),
                exposure_mean=d['x'].mean(), exposure_sd=d['x'].std(),
                largest_weight_firm=str(d['FK'].conm.iloc[j]), largest_weight_pct=100 * sk[j],
                largest_weight_ai_rev=float(d['g_lab'][j]))


def single():
    """Each event as a collapsed cross-section, for every sample and strata; writes
    shift_share_single.csv and the shift descriptives."""
    rows, desc = [], []
    for tag in ('chatgpt', 'deepseek'):
        for bal, perm in SAMPLES:
            S1.TAG, S1.CFG, S1.BAL, S1.PERM = tag, S1.CFG_ALL[tag], bal, perm
            S1.SHOCK, S1.EVENT = pd.Timestamp(S1.CFG['shock']), pd.Timestamp(S1.CFG['event'])
            S1.main()
            R = pd.read_parquet(OUT / ('shift_share_%s.parquet' % sfx(tag, bal, perm)))
            R.insert(1, 'sample', 'balanced' if bal else 'all funds')
            R.insert(2, 'strata', perm)
            rows.append(R)
        for bal in (False, True):
            desc.append(shifts(tag, bal))
    pd.concat(rows, ignore_index=True).to_csv(TABLES / 'shift_share_single.csv', index=False)
    pd.DataFrame(desc).to_csv(TABLES / 'shift_share_shifts.csv', index=False)


def two_events():
    """Both events under one permutation of the company labels; writes
    shift_share_two_shock.csv."""
    rows = []
    for perm in ('tech', 'techsize'):
        S2.BAL, S2.PERM = True, perm
        S2.main()
        R = pd.read_parquet(OUT / ('two_shock_permutation_balanced_%s.parquet' % perm))
        R.insert(0, 'strata', perm)
        rows.append(R)
    pd.concat(rows, ignore_index=True).to_csv(TABLES / 'shift_share_two_shock.csv', index=False)


def panel():
    """Permutation inference on the stacked monthly panel; writes
    shift_share_panel_permutation.csv."""
    rows = []
    for perm in ('tech', 'techsize'):
        S3.PERM = perm
        S3.main()
        R = pd.read_parquet(OUT / ('panel_permutation_%s.parquet' % perm))
        R.insert(0, 'strata', perm)
        rows.append(R)
    pd.concat(rows, ignore_index=True).to_csv(TABLES / 'shift_share_panel_permutation.csv',
                                              index=False)


def leave_out():
    """The stacked design without the largest AI stocks; writes shift_share_leave_out.csv."""
    S4.main()
    pd.read_parquet(OUT / 'leave_out.parquet').to_csv(TABLES / 'shift_share_leave_out.csv',
                                                      index=False)


def recentred():
    """Recentred estimates (Borusyak and Hull); writes shift_share_recentred.csv."""
    S5.main()
    pd.read_parquet(OUT / 'recentred.parquet').to_csv(TABLES / 'shift_share_recentred.csv',
                                                      index=False)


def placebo():
    """The pre-period placebo on the panel; writes shift_share_inside_placebo.csv."""
    S6.main()
    pd.read_parquet(OUT / 'inside_placebo.parquet').to_csv(
        TABLES / 'shift_share_inside_placebo.csv', index=False)


def extra():
    """Two additions to recentred's panel permutations, written to shift_share_extra.csv.

    (a) The coefficient's permutation distribution is wide because reassigning a label as
        large as NVIDIA's to a lightly held company rescales the exposure. Two statistics
        that do not inherit the rescaling: the response per standard deviation of the
        (recentred) exposure, b * sd(x), computed for every draw; and labels permuted within
        technology x size x holding-weight quintile, where a company's holding weight is
        the sum of its sleeve weights over the fund-windows (a predetermined feature of the
        shares, so a valid assignment process).
    (b) The incomplete-share control of Borusyak, Hull and Jaravel: the share of each
        pre-event portfolio in labelled companies, interacted with post and post x sign,
        added to the stacked panel; the collapsed cross-sections of single already carry it.
    The technology x size coefficient and studentised p-values must reproduce recentred's.
    """
    from scipy import stats as _stats  # noqa: F401  (keeps the import pattern of recentred)
    captured = []
    orig = S3.sleeve_weights

    def capture(tag, uids):
        """Record the sleeve weights the panel build computes, then return them unchanged."""
        h = orig(tag, uids)
        captured.append(h)
        return h
    S3.sleeve_weights = capture
    try:
        P = S5.panel()
    finally:
        S3.sleeve_weights = orig
    # Holding weight of each company: its sleeve weights summed over the fund-windows; its
    # quintiles within technology x size define the finer strata.
    hw = pd.concat(captured)
    uids = sorted(hw.uid.unique())
    assert len(uids) == P['n_u'], 'sleeve weights do not cover every fund-window'
    firms, both = P['firms'], P['both']
    tech, qn, sic2 = S5.firm_attributes(firms)
    n_k = hw.groupby('gvkey').w.sum().reindex(firms).fillna(0).to_numpy()
    hq = np.full(len(firms), -1.0)
    hq[both] = pd.qcut(pd.Series(n_k[both]).rank(method='first'), 5, labels=False).to_numpy()
    key_hold = np.where(both, tech * 100 + qn * 10 + hq, -99)
    kinds = {'techsize': S5.strata_of('techsize', both, tech, qn, sic2),
             'techsize_hold': [np.flatnonzero(key_hold == v) for v in np.unique(key_hold) if v != -99]}
    # covered share of the pre-event portfolio, per fund-window, in uids order
    cov = {}
    for tag in ('chatgpt', 'deepseek'):
        E = pd.read_parquet(S3.EXPS[tag], columns=['series_id', 'report_date', 'filing_date', 'labelled'])
        E['report_date'] = pd.to_datetime(E.report_date)
        E['filing_date'] = pd.to_datetime(E.filing_date)
        pre = E[(E.filing_date < S3.treat_cutoff(S3.SHOCK[tag])) & (E.labelled >= 50)]
        pre = pre.sort_values('report_date').groupby('series_id').tail(1)
        cov.update({s + '|' + tag: v for s, v in zip(pre.series_id, pre.labelled)})
    cov_pre = np.array([cov[u] for u in uids])
    g1, g2 = P['g']['chatgpt'], P['g']['deepseek']
    M, sgn, Z = P['M'], P['sgn'], P['Z']
    from src.analysis.flows.estimation import ols_cluster  # noqa: E402

    def fit(x, ctls):
        """The direction coefficient and its standard error for exposure `x` and extra
        controls `ctls`."""
        xa = M @ x
        cols = [xa, sgn * xa, Z]
        for c in ctls:
            ca = M @ c
            cols += [ca[:, None], (sgn * ca)[:, None]]
        b, se, *_ = ols_cluster(P['Y'], np.column_stack(cols), P['clus'], n_absorbed=P['nabs'])
        return b[1], se[1]

    # Per strata: the observed statistics and N_PERM label permutations; the technology x
    # size results must reproduce shift_share_recentred.
    ref = pd.read_csv(TABLES / 'shift_share_recentred.csv')
    ref = ref[(ref.design == 'panel') & (ref.strata == 'techsize')].set_index('estimate')
    rows = []
    for kind, st in kinds.items():
        mu = P['ai_pre_from'](S5.expected_labels(g1, st), S5.expected_labels(g2, st))
        specs = {'raw': (P['apre'], []), 'recentred': (P['apre'] - mu, [])}
        if kind == 'techsize':
            specs.update({'raw+cov': (P['apre'], [cov_pre]), 'recentred+cov': (P['apre'] - mu, [cov_pre])})
        obs = {}
        for k, (x, ctls) in specs.items():
            b, se = fit(x, ctls)
            obs[k] = (b, b / se, b * np.std(x), se)
        rng = np.random.default_rng(S5.SEED)
        D = {k: np.empty((N_PERM, 3)) for k in specs}
        for r in range(N_PERM):
            p = S5.draw(rng, len(firms), st)
            ap = P['ai_pre_from'](g1[p], g2[p])
            for k, (x0, ctls) in specs.items():
                x = ap if k.startswith('raw') else ap - mu
                b, se = fit(x, ctls)
                D[k][r] = (b, b / se, b * np.std(x))
        for k in specs:
            b, t, bs, se = obs[k]
            pc, pt, ps = (S5.pv(b, D[k][:, 0]), S5.pv(t, D[k][:, 1]), S5.pv(bs, D[k][:, 2]))
            if kind == 'techsize' and k in ('raw', 'recentred') and N_PERM == 2000:
                assert abs(b - ref.loc[k, 'b']) < 1e-9 and abs(pc - ref.loc[k, 'p_coef']) < 1e-12 \
                    and abs(pt - ref.loc[k, 'p_t']) < 1e-12, 'extra does not reproduce recentred (%s)' % k
            rows.append(dict(strata=kind, estimate=k, b=b, se=se, t=t, b_per_sd=bs, sd_x=bs / b if b else np.nan,
                             perm_mean=D[k][:, 0].mean(), perm_sd=D[k][:, 0].std(), p_coef=pc, p_t=pt,
                             p_per_sd=ps, n_strata=len(st)))
            print('  %-14s %-14s b %+.4f (t %.2f) per-SD %+.3f  p coef %.4f  t %.4f  per-SD %.4f'
                  % (kind, k, b, t, bs, pc, pt, ps), flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(TABLES / 'shift_share_extra.csv', index=False)
    print('covered share of the pre-event portfolio: mean %.1f%%, sd %.1f; strata: %s'
          % (cov_pre.mean(), cov_pre.std(), {k: len(v) for k, v in kinds.items()}))


STAGES = {'single': single, 'two_events': two_events, 'panel': panel, 'leave_out': leave_out, 'recentred': recentred, 'placebo': placebo,
          'extra': extra}


def headline():
    """The numbers the manuscript quotes, from whichever tables exist."""
    H = {'labels': LABELS, 'n_perm': N_PERM}
    f = TABLES / 'shift_share_single.csv'
    if f.exists():
        R = pd.read_csv(f)
        H['single'] = {}
        for tag in ('chatgpt', 'deepseek'):
            r = R[(R['shock'] == tag) & (R['sample'] == 'balanced') & (R['strata'] == 'techsize')]
            g = lambda d, i: r[(r['design'] == d) & (r['inference'] == i)].iloc[0]  # noqa: E731
            m = g('A label shift', 'conventional HC1')
            H['single'][tag] = {
                'b': float(m.b), 'se_conventional': float(m.se),
                'se_akm0': float(g('A label shift', 'AKM0 SIC2').se),
                'se_exposure_robust': float(g('A label shift', 'exposure-robust SIC2').se),
                'perm_sd': float(g('A label shift', 'shift permutation sd').se),
                'placebo_b': float(g('A placebo pre-period', 'exposure-robust SIC2').b),
                'placebo_perm_sd': float(g('A placebo pre-period', 'shift permutation sd').se)}
            if tag == 'deepseek' and (r['design'] == 'B first stage').any():
                fs = g('B first stage', 'exposure-robust SIC2')
                rf = g('B price shift, reduced form', 'exposure-robust SIC2')
                iv = g('B IV, label instrumented by price', 'exposure-robust SIC2')
                H['single'][tag].update(
                    first_stage_b=float(fs.b), first_stage_F=float((fs.b / fs.se) ** 2),
                    reduced_form_b=float(rf.b), reduced_form_t=float(rf.b / rf.se),
                    iv_b=float(iv.b), iv_se=float(iv.se))
        D = pd.read_csv(TABLES / 'shift_share_shifts.csv')
        H['shifts'] = D.to_dict('records')
    f = TABLES / 'shift_share_two_shock.csv'
    if f.exists():
        R = pd.read_csv(f)
        r = R[R.strata == 'techsize'].set_index('stat')
        H['two_shock'] = {k: {c: float(r.loc[k, c]) for c in ('observed', 'perm_sd', 'p')}
                          for k in r.index}
    f = TABLES / 'shift_share_panel_permutation.csv'
    if f.exists():
        R = pd.read_csv(f)
        H['panel_permutation'] = {s: {k: float(v) for k, v in
                                      R[(R.strata == s) & (R.stat == 'b')].iloc[0].drop(['strata', 'stat']).items()}
                                  for s in ('tech', 'techsize')}
    f = TABLES / 'shift_share_recentred.csv'
    if f.exists():
        R = pd.read_csv(f)
        H['recentred'] = {}
        for design in ('panel', 'collapsed_balanced', 'collapsed_all'):
            H['recentred'][design] = {}
            for s in ('none', 'size', 'tech', 'techsize', 'sic2'):
                for est in ('raw', 'mu only', 'recentred', 'rec+mu'):
                    r = R[(R.design == design) & (R.strata == s) & (R.estimate == est)]
                    if len(r):
                        r = r.iloc[0]
                        H['recentred'][design]['%s|%s' % (s, est)] = {
                            k: (None if pd.isna(r[k]) else float(r[k]))
                            for k in ('b', 't', 'perm_mean', 'perm_sd', 'p_coef', 'p_t')}
    f = TABLES / 'shift_share_extra.csv'
    if f.exists():
        R = pd.read_csv(f)
        H['extra'] = {'%s|%s' % (r.strata, r.estimate): {k: float(r[k]) for k in
                                                         ('b', 'se', 't', 'b_per_sd', 'sd_x', 'perm_mean', 'perm_sd',
                                                          'p_coef', 'p_t', 'p_per_sd')}
                      for _, r in R.iterrows()}
    f = TABLES / 'shift_share_leave_out.csv'
    if f.exists():
        R = pd.read_csv(f)
        H['leave_out'] = {'%s|%s' % (r['sample'], r.term): {'b': float(r.b), 'se': float(r.se)}
                          for _, r in R.iterrows()}
    f = TABLES / 'shift_share_inside_placebo.csv'
    if f.exists():
        R = pd.read_csv(f)
        H['inside_placebo'] = {'%s|%s|%s|%s' % (r.controls, r.kind, r.window, r.term):
                               {'b': float(r.b), 'se': float(r.se)} for _, r in R.iterrows()}
    (TABLES / 'shift_share.json').write_text(json.dumps(H, indent=1), encoding='utf-8')
    return H


def main(argv):
    """Run the requested stages, then collect the quoted numbers in shift_share.json."""
    stages = argv or list(STAGES)
    bad = [s for s in stages if s not in STAGES and s != 'headline']
    assert not bad, 'unknown stages %s' % bad
    print('shift-share inference on the %s labels, N_PERM %d, outputs in %s'
          % (LABELS, N_PERM, OUT.parent), flush=True)
    if stages != ['headline']:                 # `headline` only re-reads the tables
        n = holdings_with_universe_sector()
        print('holdings file with the universe sector: %d rows in the universe -> %s'
              % (n, HOLD), flush=True)
    for s in [s for s in stages if s != 'headline']:
        t0 = time.time()
        STAGES[s]()
        print('\n### %s done in %.0f s\n' % (s, time.time() - t0), flush=True)
    H = headline()
    if 'recentred' in H and 'techsize|recentred' in H['recentred'].get('panel', {}):
        r = H['recentred']['panel']['techsize|recentred']
        print('panel, recentred within technology x size: b %+.4f (t %.2f), p(t) %.4f, p(coef) %.4f'
              % (r['b'], r['t'], r['p_t'], r['p_coef']))
    print('wrote', TABLES / 'shift_share.json')


if __name__ == '__main__':
    main(sys.argv[1:])

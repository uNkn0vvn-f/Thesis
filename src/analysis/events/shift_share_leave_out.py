"""The two-event panel design, unchanged, on pre-shock exposures recomputed without the
largest AI stocks - and, beside each, the same design net of the technology share.

shift_share_panel found the two-shock response b significant under shift-level (studentised) permutation
but reproduced in magnitude by the permutation mean within technology x size strata, which
says that exposure to large technology names carries much of it. This file asks which names:
each fund's pre-shock AI exposure and technology share are recomputed from its holdings with
a set of firms removed, and the design is re-estimated exactly as s02 runs it.

Recomputing with nothing left out must reproduce the exposure the design already uses; that
is checked first, on a check that cannot pass vacuously (no NaN in either side). A fund whose
only labelled holdings are the ones left out has no exposure left and leaves the design; it
does not keep its old exposure; otherwise a single-stock NVIDIA fund would ride along in
the "without NVIDIA" sample.
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
TMP = os.path.join(OUT, 'leave_out')
os.makedirs(TMP, exist_ok=True)
X = {t: os.path.join(tp.out('analysis', 'preevent', '03_exposure', 'out'), t, 'fund_quarter_exposure.parquet')
     for t in ('chatgpt', 'deepseek')}
from src.analysis.events import two_events as T2                                  # noqa: E402
from src.analysis.events.event_dates import treat_cutoff                         # noqa: E402

SHOCK = {'chatgpt': pd.Timestamp('2022-12-01'), 'deepseek': pd.Timestamp('2025-02-01')}
LEAVE = ['NVIDIA', 'MICROSOFT', 'TAIWAN SEMICONDUCTOR', 'BROADCOM', 'ALPHABET']


def load_holdings():
    H = pd.read_parquet(tp.out('holdings', 'holdings_bridged.parquet'),
                        columns=['series_id', 'report_date', 'gvkey', 'value', 'sector', 'conm'])
    H['report_date'] = pd.to_datetime(H.report_date)
    H = H[(H.value > 0) & H.gvkey.notna()].copy()
    H['gvkey'] = H.gvkey.astype(str)
    return H


def exposure_without(H, tag, excl):
    E = pd.read_parquet(X[tag])
    E['report_date'] = pd.to_datetime(E.report_date)
    E['filing_date'] = pd.to_datetime(E.filing_date)
    pre = E[(E.filing_date < treat_cutoff(SHOCK[tag])) & (E.labelled >= 50)]
    # the latest row, not .last(): .last() fills each column from the last NON-NULL value
    pre = pre.sort_values('report_date').groupby('series_id').tail(1).reset_index(drop=True)
    L = pd.read_parquet(os.path.join(tp.out('analysis', 'preevent', '02_label', 'out'), tag,
                                     'firm_ai_labels.parquet'), columns=['gvkey', 'ai_rev'])
    L['gvkey'] = L.gvkey.astype(str)
    # the exposure panel's "seen" sleeve is every labelled firm; tech is averaged over all
    # of it, ai_rev only over the firms where it exists (e01: ok = notna)
    h = H.merge(pre[['series_id', 'report_date']], on=['series_id', 'report_date'])
    h = h.merge(L, on='gvkey', how='inner')
    h = h[~h.gvkey.isin(excl)]
    h['tech'] = (h.sector == 'tech').astype(float) * 100
    tech_new = h.groupby('series_id').apply(
        lambda g: np.average(g.tech, weights=g.value), include_groups=False).rename('tech_new')
    ha = h[h.ai_rev.notna()]
    ai_new = ha.groupby('series_id').apply(
        lambda g: np.average(g.ai_rev, weights=g.value), include_groups=False).rename('ai_rev_new')
    pre = pre.merge(ai_new, left_on='series_id', right_index=True, how='left')
    pre = pre.merge(tech_new, left_on='series_id', right_index=True, how='left')
    # every pre-shock row is overwritten, INCLUDING with NaN
    E2 = E.merge(pre[['series_id', 'report_date', 'ai_rev_new', 'tech_new']].assign(_pre=1),
                 on=['series_id', 'report_date'], how='left')
    m = E2._pre.eq(1)
    E2.loc[m, 'ai_rev'] = E2.loc[m, 'ai_rev_new']
    E2.loc[m, 'tech'] = E2.loc[m, 'tech_new']
    path = os.path.join(TMP, 'exp_%s_%d.parquet' % (tag, len(excl)))
    E2.drop(columns=['_pre', 'ai_rev_new', 'tech_new']).to_parquet(path, index=False)
    return path, pre


def estimate(p1, p2):
    L = T2.panel()
    D = pd.concat([T2.window('chatgpt', SHOCK['chatgpt'], +1.0, p1, L),
                   T2.window('deepseek', SHOCK['deepseek'], -1.0, p2, L)], ignore_index=True)
    for a, b in (('ai_post', 'ai_pre'), ('cl_post', 'claim_pre'), ('tech_post', 'tech_pre')):
        D[a] = D[b] * D.post
        D[a + '_sign'] = D[b] * D.post * D.sign
    b, se, V, n, G = T2.fit(D, ['ai_post', 'ai_post_sign', 'cl_post', 'cl_post_sign'] + T2.XC,
                            None)
    out = {'n': (n, G)}                    # the estimation sample, after singletons
    out['a'], out['b'] = (b[0], se[0]), (b[1], se[1])
    b, se, *_ = T2.fit(D, ['ai_post', 'ai_post_sign', 'tech_post', 'tech_post_sign', 'cl_post',
                           'cl_post_sign'] + T2.XC, None)
    out['b_tech'] = (b[1], se[1])
    for win in ('chatgpt', 'deepseek'):
        Dw = D[D.win == win]
        b, se, *_ = T2.fit(Dw, ['ai_post', 'cl_post'] + T2.XC, None)
        out[win] = (b[0], se[0])
        b, se, *_ = T2.fit(Dw, ['ai_post', 'tech_post', 'cl_post'] + T2.XC, None)
        out[win + '_tech'] = (b[0], se[0])
    return out


def main():
    buf = io.StringIO()

    def w(s=''):
        buf.write(s + '\n')
        try:
            print(s)
        except UnicodeEncodeError:
            print(s.encode('ascii', 'replace').decode())

    H = load_holdings()
    w('=' * 104)
    w('THE TWO-EVENT PANEL WITHOUT THE LARGEST AI STOCKS, and net of the technology share')
    w('=' * 104)
    for tag in ('chatgpt', 'deepseek'):
        p, pre = exposure_without(H, tag, set())
        m = pre.ai_rev.notna()
        d = (pre.ai_rev - pre.ai_rev_new).abs()[m]
        dt = (pre.tech - pre.tech_new).abs()[m]
        assert d.notna().all() and dt.notna().all(), 'NaN in the check - it would pass vacuously'
        assert d.max() < 1e-6 and dt.max() < 1e-6, 'recomputation does not reproduce the exposure file'
        w('  recomputation check, %-8s ai_rev max |diff| %.1e, tech max |diff| %.1e, %d funds'
          % (tag, d.max(), dt.max(), int(m.sum())))
    names = H.drop_duplicates('gvkey').set_index('gvkey').conm.astype(str).str.upper()
    gv = []
    for o in LEAVE:
        cand = names[names.str.startswith(o)]
        assert len(cand) == 1, '%s matches %d firms' % (o, len(cand))
        gv.append(cand.index[0])
        w('  leave-out firm: %-24s gvkey %s  (%s)' % (o, cand.index[0], cand.iloc[0]))
    w()
    sets = [('nothing left out', set()), ('without NVIDIA', {gv[0]}),
            ('without Microsoft', {gv[1]}), ('without the top 5', set(gv))]
    f = lambda v: '%+.4f (t %5.2f)' % (v[0], v[0] / v[1])
    res, rows = {}, []
    for lab, ex in sets:
        p1, _ = exposure_without(H, 'chatgpt', ex)
        p2, _ = exposure_without(H, 'deepseek', ex)
        res[lab] = estimate(p1, p2)
        for k, v in res[lab].items():
            if k != 'n':
                rows.append(dict(sample=lab, term=k, b=v[0], se=v[1]))
    w('  %-20s %9s %6s %20s %20s %20s %20s' % ('', 'fund-mo', 'funds', 'ChatGPT', 'DeepSeek',
                                                'a', 'b (two shocks)'))
    for lab, o in res.items():
        w('  %-20s %9d %6d %20s %20s %20s %20s' % (lab, o['n'][0], o['n'][1], f(o['chatgpt']),
                                                    f(o['deepseek']), f(o['a']), f(o['b'])))
    w()
    w('  NET OF TECHNOLOGY (each fund\'s technology share, with its own post terms)')
    w('  %-20s %9s %6s %20s %20s %20s' % ('', '', '', 'ChatGPT', 'DeepSeek', 'b (two shocks)'))
    for lab, o in res.items():
        w('  %-20s %9s %6s %20s %20s %20s' % (lab, '', '', f(o['chatgpt_tech']),
                                               f(o['deepseek_tech']), f(o['b_tech'])))
    pd.DataFrame(rows).to_parquet(os.path.join(OUT, 'leave_out.parquet'), index=False)
    w()
    w('wrote out/leave_out.parquet')
    with io.open(os.path.join(LOGS, 'shift_share_leave_out.txt'), 'w', encoding='utf-8') as fh:
        fh.write(buf.getvalue())


if __name__ == '__main__':
    main()

"""The pre-period placebo, asked of the panel design itself.

The collapsed cross-section of shift_share fails a placebo inside DeepSeek's pre-period: the change
in mean flow from months -12..-7 to months -6..-1 loads on AI exposure at -0.154 (t -3.40),
as large as the post-shock slope, and it does so under every construction of the exposure
and controls tried. The two-event panel reports a flat pre-shock slope for the same window
(-0.001 per month, t -0.23). If the panel passed only because it was never asked the
collapsed version's question, the panel's result would inherit the failure.

So the same question is asked of the panel. Each window is cut to its twelve pre-shock
months, month -6 is treated as the shock, and the two_events regression is run unchanged:
fund-within-window and window x month effects, the same controls, AI x post and, stacked,
AI x post x sign. Three control sets show which of the panel's ingredients matters:

    full        log size, own return, lagged flow       (two_events)
    no lag      log size, own return
    none        fixed effects only

and the real design is run alongside with the same three sets, so every placebo sits next to
the estimate it is a placebo for.
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
from src.analysis.flows.estimation import absorb, make_dims, absorbed_k               # noqa: E402
from src.analysis.flows.estimation import ols_cluster                               # noqa: E402
SHOCK = {'chatgpt': pd.Timestamp('2022-12-01'), 'deepseek': pd.Timestamp('2025-02-01')}
SIGN = {'chatgpt': +1.0, 'deepseek': -1.0}
CONTROLS = {'full': ['lsize', 'ret', 'nf_lag_w'], 'no lag': ['lsize', 'ret'], 'none': []}


def estimate(D, xn):
    """The two_events regression on `D` with regressors `xn`: coefficients, standard errors,
    observations and funds."""
    sub, dims = make_dims(D, [['uid'], ['win', 'month']])
    Y = absorb(sub[['nf_w']].values, dims)[:, 0]
    X = absorb(sub[xn].values, dims)
    b, se, *_ = ols_cluster(Y, X, sub.series_id.values,
                            n_absorbed=absorbed_k(dims, sub.series_id.values))
    return b, se, len(sub), sub.series_id.nunique()


def main():
    """Estimate the placebo and the real design for each control set, per window and
    stacked."""
    buf = io.StringIO()

    def w(s=''):
        buf.write(s + '\n')
        print(s)

    L = T2.panel()
    W = {t: T2.window(t, SHOCK[t], SIGN[t], EXPS[t], L) for t in ('chatgpt', 'deepseek')}
    rows = []
    w('=' * 96)
    w('THE PRE-PERIOD PLACEBO ON THE PANEL: months -12..-1, month -6 treated as the shock')
    w('=' * 96)
    w('  %-8s %-9s %-9s %22s %22s' % ('', 'controls', 'window', 'placebo: AI x post',
                                     'real: AI x post'))
    for cset, xc in CONTROLS.items():
        stacked = {}
        for kind in ('placebo', 'real'):
            parts = []
            for t, D in W.items():
                D = D.copy()
                # The placebo keeps the twelve pre-event months and moves the event to month
                # -6.
                if kind == 'placebo':
                    D = D[D.month < SHOCK[t]].copy()
                    D['post'] = (D.month >= SHOCK[t] - pd.DateOffset(months=6)).astype(float)
                D['ai_post'] = D.ai_pre * D.post
                D['ai_post_sign'] = D.ai_post * D.sign
                D['cl_post'] = D.claim_pre * D.post
                D['cl_post_sign'] = D.cl_post * D.sign
                parts.append(D)
                b, se, n, nf = estimate(D, ['ai_post', 'cl_post'] + xc)
                rows.append(dict(controls=cset, kind=kind, window=t, term='ai_post', b=b[0],
                                 se=se[0], n=n, funds=nf))
            S = pd.concat(parts, ignore_index=True)
            b, se, n, nf = estimate(S, ['ai_post', 'ai_post_sign', 'cl_post', 'cl_post_sign'] + xc)
            for j, term in ((0, 'a'), (1, 'b')):
                rows.append(dict(controls=cset, kind=kind, window='stacked', term=term, b=b[j],
                                 se=se[j], n=n, funds=nf))
            stacked[kind] = (b[1], se[1])
        # Report placebo and real estimates side by side.
        for t in ('chatgpt', 'deepseek'):
            p = [r for r in rows if r['controls'] == cset and r['window'] == t]
            pl = next(r for r in p if r['kind'] == 'placebo')
            re_ = next(r for r in p if r['kind'] == 'real')
            w('  %-8s %-9s %-9s %+12.4f (t %+5.2f) %+12.4f (t %+5.2f)'
              % ('', cset, t, pl['b'], pl['b'] / pl['se'], re_['b'], re_['b'] / re_['se']))
        (bp, sp), (br, sr) = stacked['placebo'], stacked['real']
        w('  %-8s %-9s %-9s %+12.4f (t %+5.2f) %+12.4f (t %+5.2f)'
          % ('', cset, 'b', bp, bp / sp, br, br / sr))
        w()
    R = pd.DataFrame(rows)
    R.to_parquet(os.path.join(OUT, 'inside_placebo.parquet'), index=False)
    w('wrote out/inside_placebo.parquet')
    with io.open(os.path.join(LOGS, 'shift_share_placebo.txt'), 'w', encoding='utf-8') as fh:
        fh.write(buf.getvalue())


if __name__ == '__main__':
    main()

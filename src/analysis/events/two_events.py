"""Stacked design for the two events of opposite sign.

ChatGPT (2022-11-30) raised demand for AI; DeepSeek R1 (2025-01-27) cut AI valuations
sharply. Stacking the two windows asks in one regression whether the flow response to AI
holdings follows the direction of the news:

    nf_it = a  (AI_pre x post)                       a standing AI-fund premium
          + b  (AI_pre x post x sign)                a response to the shock's direction
          + controls + fund-within-window FE + window x month FE

sign = +1 in the ChatGPT window and -1 in the DeepSeek window. Each window uses its own
pre-event treatment, the holdings filed before that event, so a fund can be treated
differently in the two windows; fund effects are fund-within-window for the same reason.

    panel    the monthly flow sample with lagged flow and winsorised rates
    window   one event window with its pre-event AI, technology and claim exposures
    fit      the stacked regression, fund-clustered
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from src import paths as tp  # noqa: E402

import numpy as np
import pandas as pd

from src.analysis.flows.estimation import ols_cluster, MIN_TNA          # noqa: E402
from src.analysis.flows.estimation import absorb, make_dims, absorbed_k    # noqa: E402
from src.analysis.events.event_dates import treat_cutoff                         # noqa: E402

LAD = tp.out('analysis', 'flow_ladder', 'out', 'fund_month_ladder.parquet')
WIN = 12
XC = ['lsize', 'ret', 'nf_lag_w']


def panel():
    L = pd.read_parquet(LAD, columns=['series_id', 'month', 'nf_beg', 'na_beg_recon',
                                      'ret'])
    L['month'] = pd.to_datetime(L.month)
    L['nf'] = L.nf_beg.replace([np.inf, -np.inf], np.nan)
    L = L[L.na_beg_recon.notna() & (L.na_beg_recon >= MIN_TNA) & L.nf.notna()
          & L.ret.notna()].sort_values(['series_id', 'month'])
    L['nf_lag'] = L.groupby('series_id').nf.shift()
    L['lsize'] = np.log(L.na_beg_recon)
    L = L[L.nf_lag.notna()]
    for c in ('nf', 'nf_lag'):
        lo = L.groupby('month')[c].transform(lambda s: s.quantile(.01))
        hi = L.groupby('month')[c].transform(lambda s: s.quantile(.99))
        L[c + '_w'] = L[c].clip(lo, hi)
    return L


def window(name, shock, sign, exp, L):
    E = pd.read_parquet(exp)
    E['report_date'] = pd.to_datetime(E.report_date)
    E['filing_date'] = pd.to_datetime(E.filing_date)
    for c in ('uses_ai', 'theme_broad'):
        E[c] = E[c].astype('boolean').fillna(False).astype(bool)
    cut = treat_cutoff(shock)                 # filed strictly before the event itself
    pre = E[(E.filing_date < cut) & (E.labelled >= 50)]
    # the latest row, not .last(): .last() fills each column from the last NON-NULL value,
    # so a missing field would silently come from an earlier quarter
    pre = pre.sort_values('report_date').groupby('series_id').tail(1).set_index('series_id')
    assert pre.filing_date.max() < cut
    # `tech_pre` rides along so the design can hold the SECTOR fixed. A fund holding AI
    # holds technology, and the sign flip rules that out only JOINTLY, across the two
    # windows; a reader of one shock is entitled to see it ruled out inside that shock.
    # The share is built from SIC through the universe's `sector` and carries no LLM
    # label, so it cannot inherit an error from the treatment it is controlling.
    tr = pd.DataFrame({'ai_pre': pre.ai_rev,
                       'tech_pre': pre.tech,
                       'claim_pre': (pre.uses_ai | pre.theme_broad).astype(float)})
    lo, hi = shock - pd.DateOffset(months=WIN), shock + pd.DateOffset(months=WIN)
    D = L[(L.month >= lo) & (L.month < hi)].merge(tr.reset_index(), on='series_id',
                                                  how='inner')
    D['post'] = (D.month >= shock).astype(float)
    D['sign'] = sign
    D['win'] = name
    D['uid'] = D.series_id + '|' + name
    need = ['ai_pre', 'tech_pre', 'claim_pre', 'nf_w'] + XC
    D = D[np.isfinite(D[need].to_numpy(dtype=float)).all(axis=1)].copy()
    return D


def fit(D, xn, w):
    sub, dims = make_dims(D, [['uid'], ['win', 'month']])
    Y = absorb(sub[['nf_w']].values, dims)[:, 0]
    X = absorb(sub[xn].values, dims)
    A = np.asarray(X, float)
    assert np.isfinite(A).all() and (A.std(axis=0) > 1e-12).all(), 'dead regressor'
    b, se, V, G = ols_cluster(Y, X, sub.series_id.values,
                              n_absorbed=absorbed_k(dims, sub.series_id.values))
    return b, se, V, len(sub), G

"""The stacked design with NON-AI technology held fixed, on the real pair of events and
on placebo pairs.

The fund's whole technology share is a weak control, because the AI firms are technology
firms and the two shares are strongly correlated. The sharper control is technology
without AI: the value share of technology firms that the window's own pre-event label
scores at zero AI sales. A fund's AI exposure and its non-AI technology exposure can then
move separately, and the question "is it AI or is it the sector?" is asked of the part of
the sector that is not AI.

The estimator is that of two_events: fund-within-window and window x month effects,
AI x post (a) and AI x post x sign (b), the claim's two terms and the control's two terms.
`exposure_file` replaces `tech` on each fund's pre-event portfolio row by the non-AI
technology share, computed over the labelled sleeve as src.holdings.ai_exposure computes
`tech`; with the AI firms kept in, the recomputation reproduces the stored `tech`.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from src import paths as tp  # noqa: E402

import numpy as np
import pandas as pd

TMP = tp.out('analysis', 'events', 'nonai_technology')
EX = {t: os.path.join(tp.out('analysis', 'preevent', '03_exposure', 'out'), t, 'fund_quarter_exposure.parquet')
      for t in ('chatgpt', 'deepseek')}
LAB = {t: os.path.join(tp.out('analysis', 'preevent', '02_label', 'out'), t, 'firm_ai_labels.parquet')
       for t in ('chatgpt', 'deepseek')}
from src.analysis.events import two_events as S2                                   # noqa: E402
from src.analysis.events.event_dates import treat_cutoff                          # noqa: E402

PAIRS = [('2022-12-01', '2025-02-01', 'REAL'),
         ('2021-12-01', '2024-01-01', 'placebo, a year before each'),
         ('2021-06-01', '2023-12-01', 'placebo'),
         ('2021-06-01', '2024-01-01', 'placebo'),
         ('2021-12-01', '2023-12-01', 'placebo')]


def holdings():
    H = pd.read_parquet(tp.out('holdings', 'holdings_bridged.parquet'),
                        columns=['series_id', 'report_date', 'gvkey', 'value', 'sector'])
    H['report_date'] = pd.to_datetime(H.report_date)
    H = H[(H.value > 0) & H.gvkey.notna()].copy()
    H['gvkey'] = H.gvkey.astype(str)
    return H


def exposure_file(H, tag, date, excl=frozenset()):
    """The window's exposure file with `tech` replaced, on each fund's pre-date row, by the
    non-AI technology share. With `excl`, those firms are removed from the holdings and the
    AI exposure is recomputed without them too (shift_share_leave_out's rule: a fund with nothing labelled
    left leaves the design). Returns the path and the recomputation check."""
    os.makedirs(TMP, exist_ok=True)
    date = pd.Timestamp(date)
    E = pd.read_parquet(EX[tag])
    E['report_date'] = pd.to_datetime(E.report_date)
    E['filing_date'] = pd.to_datetime(E.filing_date)
    pre = E[(E.filing_date < treat_cutoff(date)) & (E.labelled >= 50)]
    pre = pre.sort_values('report_date').groupby('series_id').tail(1)
    L = pd.read_parquet(LAB[tag], columns=['gvkey', 'ai_rev'])
    L['gvkey'] = L.gvkey.astype(str)
    h = H.merge(pre[['series_id', 'report_date']], on=['series_id', 'report_date'])
    h = h.merge(L, on='gvkey', how='inner')                  # the labelled sleeve
    h = h[~h.gvkey.isin(excl)]
    h['tech'] = (h.sector == 'tech').astype(float) * 100
    h['nonai_tech'] = ((h.sector == 'tech') & (h.ai_rev.fillna(0) == 0)).astype(float) * 100
    g = h.groupby('series_id')
    ha = h[h.ai_rev.notna()]
    share = pd.DataFrame({
        'tech_chk': g.apply(lambda d: np.average(d.tech, weights=d.value), include_groups=False),
        'nonai': g.apply(lambda d: np.average(d.nonai_tech, weights=d.value),
                         include_groups=False),
        'ai_new': ha.groupby('series_id').apply(lambda d: np.average(d.ai_rev, weights=d.value),
                                                include_groups=False)})
    pre = pre.merge(share, left_on='series_id', right_index=True, how='left')
    diff = (pre.tech - pre.tech_chk).abs()
    if not excl:
        d_ai = (pre.ai_rev - pre.ai_new).abs()[pre.ai_rev.notna()]
        assert d_ai.notna().all() and d_ai.max() < 1e-6, 'ai_rev recomputation fails'
    E2 = E.merge(pre[['series_id', 'report_date', 'nonai', 'ai_new']].assign(_p=1),
                 on=['series_id', 'report_date'], how='left')
    m = E2._p.eq(1)
    E2.loc[m, 'tech'] = E2.loc[m, 'nonai']
    E2.loc[m, 'ai_rev'] = E2.loc[m, 'ai_new']              # NaN where nothing labelled is left
    path = os.path.join(TMP, 'exp_%s_%s_%d.parquet' % (tag, date.date(), len(excl)))
    E2.drop(columns=['_p', 'nonai', 'ai_new']).to_parquet(path, index=False)
    corr = pre[['ai_rev', 'tech', 'nonai']].astype(float).corr()
    return path, float(diff.max()), corr


def stacked(L2, p1, d1, p2, d2):
    D = pd.concat([S2.window('chatgpt', pd.Timestamp(d1), +1.0, p1, L2),
                   S2.window('deepseek', pd.Timestamp(d2), -1.0, p2, L2)], ignore_index=True)
    for a, b in (('ai_post', 'ai_pre'), ('cl_post', 'claim_pre'), ('tech_post', 'tech_pre')):
        D[a] = D[b] * D.post
        D[a + '_sign'] = D[b] * D.post * D.sign
    b, se, V, n, G = S2.fit(D, ['ai_post', 'ai_post_sign', 'tech_post', 'tech_post_sign',
                                'cl_post', 'cl_post_sign'] + S2.XC, None)
    out = {k: (b[i], se[i]) for i, k in enumerate(['a', 'b', 'nt', 'nt_sign'])}
    for win in ('chatgpt', 'deepseek'):
        bb, ss, *_ = S2.fit(D[D.win == win], ['ai_post', 'tech_post', 'cl_post'] + S2.XC, None)
        out[win] = (bb[0], ss[0])
    return out, n, G

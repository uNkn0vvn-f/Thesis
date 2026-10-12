"""Event-window sample for a single dated event.

`build` takes each fund's latest portfolio filed before the event (event_dates.treat_cutoff)
with at least half its value labelled, and merges its AI revenue share and AI claim onto the
monthly flow panel within WIN months either side of the event. The pre-event joint tests of
src/analysis/robustness/event_pretrends.py estimate on this sample.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))

import numpy as np
import pandas as pd

from src.analysis.events.event_dates import treat_cutoff                         # noqa: E402

WIN = 12                                 # months either side of the event
XC = ['lsize', 'ret', 'nf_lag_w']


def build(shock, E, L, thematic):
    # FILING date, not report date. Among portfolios public before
    # the EVENT (event_dates.py), take the one with the latest report date.
    pre = E[(E.filing_date < treat_cutoff(shock)) & (E.labelled >= 50)]
    # the latest row, not .last(): .last() fills each column from the last NON-NULL value,
    # so a missing field would silently come from an earlier quarter
    pre = pre.sort_values('report_date').groupby('series_id').tail(1).set_index('series_id')
    tr = pd.DataFrame({'ai_pre': pre.ai_rev,
                       'claim_pre': (pre.uses_ai | pre.theme_broad).astype(float),
                       'ai_date': pre.report_date})
    lo, hi = shock - pd.DateOffset(months=WIN), shock + pd.DateOffset(months=WIN)
    D = L[(L.month >= lo) & (L.month < hi)].reset_index(drop=True).copy()
    D = D.merge(tr.reset_index(), on='series_id', how='inner')
    D['k'] = (D.month.dt.year - shock.year) * 12 + (D.month.dt.month - shock.month)
    D['thematic'] = D.series_id.isin(thematic)
    need = ['ai_pre', 'claim_pre', 'nf_w', 'lsize', 'ret', 'nf_lag_w']
    bad = ~np.isfinite(D[need].to_numpy(dtype=float)).all(axis=1)
    D = D[~bad].copy()
    D.attrs['dropped_nonfinite'] = int(bad.sum())
    D.attrs['latest_ai_date'] = pre.report_date.max()
    D.attrs['latest_filing'] = pre.filing_date.max()
    return D

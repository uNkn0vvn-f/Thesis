"""One rule for when a prospectus stops describing a fund.

Every as-of match in the repository - a fund-quarter or fund-month paired with the last
prospectus filed on or before it - carried that prospectus forward indefinitely. If the
fund's next prospectus was missed (the segmenter could not split a multi-fund filing, say),
its old claim stood in for years: ROBO Global Artificial Intelligence ETF was scored
against its May 2020 prospectus into 2024.

Section 10(a)(3) of the Securities Act bounds this from outside the data. A prospectus used
more than nine months after the registration statement became effective must carry
information no more than sixteen months old, which is why open-end funds update every year.
So a match more than sixteen months old is not the prospectus in force - it is a gap in the
corpus - and the claim for that date is UNKNOWN. It is blanked exactly as if no prospectus
had been matched, so each consumer's existing no-prospectus handling applies: the ladder's
rung -1, the exposure panel's "no claim public by the report date".

The threshold is the statute's, not a tuned one: 16 months = 487 days (16 x 30.44).
"""
import os

import numpy as np
import pandas as pd

# ASOF_MAX_AGE_DAYS overrides the statute for a sensitivity run only: a very large value
# carries every prospectus forward indefinitely
MAX_AGE_DAYS = int(os.environ.get('ASOF_MAX_AGE_DAYS', '487'))


def expire(A, at, cols, date='date'):
    """Blank the matched prospectus columns where it is older than the statute allows.

    A     the frame returned by merge_asof
    at    the column holding the date the match is made AT (report date, month, ...)
    cols  the columns that came from the prospectus side, including `date`
    Returns (A, number of rows expired)."""
    age = (pd.to_datetime(A[at]) - pd.to_datetime(A[date])).dt.days
    stale = (age > MAX_AGE_DAYS).fillna(False).to_numpy()
    n = int(stale.sum())
    if n:
        for c in cols:
            if c == date:
                A.loc[stale, c] = pd.NaT
            else:
                A[c] = A[c].astype(object)
                A.loc[stale, c] = np.nan
    return A, n

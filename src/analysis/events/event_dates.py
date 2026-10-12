"""When each shock became public, for choosing which holdings a treatment may use.

The flow panels treat the first full month after an event as `post` (2022-12, 2025-02). A
treatment that used every portfolio FILED before that month would let in filings made on or
after the event itself: 74 of the 512 DeepSeek funds' portfolios were filed on 27 January
2025 or later. The portfolios are earlier reports, so no fund traded on the news, but a
treatment built from them was not public before it.

The cutoff is therefore the event date, and a filing made ON that day is excluded, because its
time of day relative to the news is unknown:

    ChatGPT     released 30 November 2022
    DeepSeek    R1 released 20 January 2025; the market repriced AI exposure on
                27 January, the event the paper's price test and the DeepSeek treatment
                are dated to

Placebo dates carry no event and keep their own date.
"""
import pandas as pd

EVENT = {pd.Timestamp('2022-12-01'): pd.Timestamp('2022-11-30'),
         pd.Timestamp('2025-02-01'): pd.Timestamp('2025-01-27')}


def treat_cutoff(shock):
    """The date a portfolio must have been filed strictly before to enter a treatment
    whose first post month is `shock`."""
    shock = pd.Timestamp(shock)
    return EVENT.get(shock, shock)

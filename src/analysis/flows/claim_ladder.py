"""The verifiability ladder.

The ladder orders claims by how much a reader with public data could do with the
sentence. Its top two rungs are decided FUND BY FUND, from the fund's own holdings: a theme
claim that names positions is confirmed in a month if the fund's latest portfolio public
before that month holds more of at least one named position than its size-decile x sector
benchmark (abnormal exposure > 0, src.holdings.ai_exposure).

WHY FUND BY FUND. Setting rungs 4 and 5 by what the claimant GROUP holds would put a fund
at rung 5 whatever the fund itself holds, so "a claim the portfolio confirms" would not be
what the rung measures.

    rung 0   no AI claim, no AI in the name                      the control
    rung 1   AI in the NAME, no affirmative claim in the text     the cheapest claim
    rung 2   `uses_ai` - an algorithm picks the securities        process claim, not
                                                                 assessed by the company-
                                                                 exposure holdings measure
    rung 3   AI theme, no value-chain position named              nothing named to check
    rung 4   AI theme, names positions; the fund's own holdings    named, NOT confirmed
             bear out none of them
    rung 5   AI theme, names positions; the fund's own holdings    named and CONFIRMED
             bear out at least one
    rung -1  AI theme, names positions, but no qualifying holdings  UNKNOWN: neither
             public within the last twelve months                   treatment nor control

PRE-DECLARED RULES. A theme claim outranks a process claim, because the theme claim is the
checkable one. Confirming any one named position puts a fund at rung 5 even if it also names
others its holdings do not bear out. The holdings used are the latest fund-quarter FILED
before the month (strictly), dated within twelve months of it, with at least half its value
labelled; the label is the point-in-time vintage of that quarter. A claim that cannot be
checked this way is unknown, not unconfirmed.

AS-OF, NEVER `ever`. Every rung is the one implied by the most recent prospectus PUBLIC
at that month. 585 funds ever make a theme claim; 68 had made one as of ChatGPT, so an
`ever` column leaks six years of the future into a monthly design.

NOT A LADDER OF QUALITY. A higher rung is not a better fund. The company-exposure
holdings measure used here does not assess how a manager makes investment decisions.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from src import paths as tp  # noqa: E402
import io, os

import numpy as np
import pandas as pd
from src.fund_claims.asof import expire, MAX_AGE_DAYS                        # noqa: E402
from src.analysis.flows.claim_rungs import latest_disclosed_holdings                          # noqa: E402

OUT = tp.out('analysis', 'flow_ladder', 'out')
LOGS = tp.out('analysis', 'flow_ladder', 'logs')
os.makedirs(OUT, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)

FLOWS = tp.out('analysis', 'fund_flows', 'out', 'fund_month_flows.parquet')
PANEL = (tp.out('fund_claims', 'panel_fund_filing.parquet'))
SPELL = (tp.out('fund_claims', 'ai_name_spells.parquet'))

SHOCK = pd.Timestamp('2022-11-30')
EXPO = (tp.out('exposure', 'fund_quarter_exposure.parquet'))
POS = ('developer', 'enabler', 'adopter')
FRESH_DAYS = 366        # holdings dated more than a year before the month do not verify it


RUNG_NAME = {0: 'no claim', 1: 'name only', 2: 'uses AI (process)',
             3: 'theme, nothing named', 4: 'theme, named NOT confirmed',
             5: 'theme, named and CONFIRMED'}


def verify_by_fund(A, w):
    """Replace the provisional rung 5 ('theme, names a position') by the fund's own test."""
    E = pd.read_parquet(EXPO, columns=['series_id', 'report_date', 'filing_date', 'labelled']
                        + ['abn_' + p for p in POS])
    E['report_date'] = pd.to_datetime(E.report_date)
    E['filing_date'] = pd.to_datetime(E.filing_date)
    E = E[E.labelled >= 50].dropna(subset=['filing_date'])
    E = latest_disclosed_holdings(E, ['abn_' + p for p in POS])
    named = A.rung == 5
    X = A.loc[named, ['series_id', 'month'] + ['chain_' + p for p in POS]].copy()
    X['_i'] = X.index
    X = pd.merge_asof(X.sort_values('month'), E, left_on='month', right_on='filing_date',
                      by='series_id', direction='backward', allow_exact_matches=False)
    fresh = X.report_date.notna() & (X.month - X.report_date).dt.days.between(0, FRESH_DAYS)
    ok = pd.Series(False, index=X.index)
    ok5 = pd.Series(False, index=X.index)
    seen = pd.Series(False, index=X.index)
    for p in POS:
        has = X['chain_' + p] & X['abn_' + p].notna()
        seen |= has
        ok |= has & (X['abn_' + p] > 0)
        ok5 |= has & (X['abn_' + p] > 5)       # a stricter bar, reported only
    X['verif'] = np.where(~fresh | ~seen, 'unobserved', np.where(ok, 'confirmed', 'not confirmed'))
    X = X.set_index('_i')
    A['verif'] = None
    A.loc[X.index, 'verif'] = X.verif
    A.loc[X.index, 'verif_report_date'] = X.report_date
    A.loc[named & (A.verif == 'not confirmed'), 'rung'] = 4
    unk = named & (A.verif == 'unobserved')
    A.loc[unk, 'rung'] = -1
    A.loc[unk, 'rung_known'] = False
    A['verif_unknown'] = unk
    g = A[named]
    w('  theme claims naming a position, verified fund by fund against own holdings:')
    for v in ('confirmed', 'not confirmed', 'unobserved'):
        s = g[g.verif == v]
        w('    %-14s %8d fund-months %6d funds' % (v, len(s), s.series_id.nunique()))
    obs = fresh & seen
    w('    under a stricter bar (abnormal exposure above 5pp) %d of the %d observed fund-months'
      ' would be confirmed' % (int((ok5 & obs).sum()), int(obs.sum())))
    w('    ("unobserved": no portfolio with half its value labelled, filed before the month')
    w('     and dated within %d days; those fund-months are UNKNOWN, rung -1)' % FRESH_DAYS)
    return A


def main():
    buf = io.StringIO()

    def w(s=''):
        buf.write(s + '\n')
        try:
            print(s)
        except UnicodeEncodeError:
            print(s.encode('ascii', 'replace').decode())

    SP = pd.read_parquet(SPELL)
    SP['date_from'] = pd.to_datetime(SP.date_from)
    SP['date_to'] = pd.to_datetime(SP.date_to)

    P = pd.read_parquet(PANEL)
    P['date'] = pd.to_datetime(P.date, errors='coerce')
    P = P[P.series_resolved & P.series_id.notna() & P.date.notna()].copy()
    B = ['uses_ai', 'theme_broad', 'theme_strict', 'falsifiable_test',
         'chain_developer', 'chain_enabler', 'chain_adopter', 'chain_benefit_vague',
         'chain_unspecified']
    for c in B:
        P[c] = P[c].astype('boolean').fillna(False).astype(bool)

    # provisional: 5 = a theme claim naming any position; verified month by month below
    named_any = P[['chain_' + p for p in POS]].any(axis=1)
    P['rung'] = np.select([P.theme_broad & named_any, P.theme_broad, P.uses_ai], [5, 3, 2],
                          default=0)
    conflict = P.groupby(['series_id', 'date'])[B + ['cik']].nunique(dropna=False)
    if conflict.gt(1).any(axis=1).any():
        raise ValueError('Conflicting claim values for the same fund and prospectus date.')
    P = (P.sort_values(['series_id', 'date', 'accession'], kind='mergesort')
         .drop_duplicates(['series_id', 'date'], keep='last'))

    w('=' * 96)
    w('THE VERIFIABILITY LADDER')
    w('=' * 96)
    w('  rungs 4 and 5 are set fund by fund from each fund\'s own holdings (below).')
    w()
    w('  filing-level rung before verification (rung 1 is a NAME fact, applied below;')
    w('  "5" here means a theme naming a position, confirmed or not):')
    w('  %-6s %-30s %10s %10s' % ('rung', 'meaning', 'filings', 'series'))
    for r in range(6):
        g = P[P.rung == r]
        w('  %-6d %-30s %10d %10d' % (r, RUNG_NAME[r], len(g), g.series_id.nunique()))
    w()

    F = pd.read_parquet(FLOWS, columns=['series_id', 'month', 'q', 'flow', 'na_qend',
                                        'na_beg_recon', 'nf_beg', 'ret'])
    F['month'] = pd.to_datetime(F.month)
    F = F.sort_values('month')
    keep = ['series_id', 'date', 'rung', 'cik'] + B
    A = pd.merge_asof(F, P[keep].sort_values('date'), left_on='month', right_on='date',
                      by='series_id', direction='backward')
    # a prospectus more than 16 months old is not the one in force (asof.py): rung -1
    A, n_expired = expire(A, 'month', [c for c in keep if c != 'series_id'])
    A['rung_known'] = A.date.notna()
    A['claim_age_days'] = (A.month - A.date).dt.days

    S = SP.sort_values('date_from')
    A = A.sort_values('month')
    A = pd.merge_asof(A, S[['series_id', 'date_from', 'date_to', 'ai', 'series_name',
                            'cik', 'registrant_name']]
                      .rename(columns={'ai': 'ai_named', 'cik': 'cik_name'})
                      .sort_values('date_from'),
                      left_on='month', right_on='date_from', by='series_id',
                      direction='backward')
    in_force = A.date_to.isna() | (A.month < A.date_to)
    A['ai_named'] = (A.ai_named.astype('boolean').fillna(False).astype(bool)
                     & in_force)
    A.loc[(A.rung == 0) & A.ai_named & A.rung_known, 'rung'] = 1
    A['rung'] = pd.to_numeric(A.rung).fillna(-1).astype(int)
    for c in B:
        A[c] = A[c].astype('boolean').fillna(False).astype(bool)
    A = verify_by_fund(A, w)
    A['sponsor'] = A.cik_name.fillna(A.cik)

    K = A[A.rung_known]
    at_shock = K[(K.month >= SHOCK - pd.Timedelta(days=31)) & (K.month <= SHOCK)]
    w('-' * 96)
    w('THE FUND-MONTH GRID')
    w('-' * 96)
    w('  fund-months %d over %d funds, %s .. %s'
      % (len(A), A.series_id.nunique(), A.month.min().date(), A.month.max().date()))
    w('  with a prospectus public by that month : %.1f%% (median age %.0f days)'
      % (100 * A.rung_known.mean(), A.claim_age_days.median()))
    w('  ... matches older than %d days treated as unknown : %d fund-months'
      % (MAX_AGE_DAYS, n_expired))
    w('  sponsor identified for %.1f%% of fund-months, %d sponsors'
      % (100 * A.sponsor.notna().mean(), A.sponsor.nunique()))
    w()
    w('  %-6s %-30s %12s %9s %12s'
      % ('rung', 'meaning', 'fund-months', 'funds', 'as of ChatGPT'))
    for r in range(6):
        g = K[K.rung == r]
        w('  %-6d %-30s %12d %9d %12d'
          % (r, RUNG_NAME[r], len(g), g.series_id.nunique(),
             at_shock[at_shock.rung == r].series_id.nunique()))
    w()
    w('  Fund-months before a fund\'s first classified prospectus carry rung = -1 and')
    w('  are never folded into rung 0: "not yet seen to claim" and "claims nothing" are')
    w('  different statements and only the second is a control.')
    w()

    w('-' * 96)
    w('DO FUNDS MOVE?')
    w('-' * 96)
    ch = K.groupby('series_id').rung.nunique() > 1
    mv = pd.DataFrame({'first': K.groupby('series_id').rung.first(),
                       'last': K.groupby('series_id').rung.last()})
    w('  series whose rung ever changes : %d of %d (%.1f%%); climbed %d, fell %d'
      % (int(ch.sum()), len(ch), 100 * ch.mean(),
         int((mv['last'] > mv['first']).sum()), int((mv['last'] < mv['first']).sum())))
    w('  Claims are adopted and almost never dropped, which is exactly why an `ever`')
    w('  column would backdate every adoption to the start of the panel.')
    w()
    w('  AI-named funds with flow data and a prospectus : %d ; of those at rung 1'
      % K[K.ai_named].series_id.nunique())
    w('  (name says AI, text claims nothing) : %d'
      % K[(K.rung == 1)].series_id.nunique())
    w('  Name-only and text-based claims remain distinct categories.')
    w()

    A.to_parquet(os.path.join(OUT, 'fund_month_ladder.parquet'), index=False)
    w('wrote out/fund_month_ladder.parquet (%d fund-months)' % len(A))

    with io.open(os.path.join(LOGS, 'claim_ladder.txt'), 'w', encoding='utf-8') as fh:
        fh.write(buf.getvalue())


if __name__ == '__main__':
    main()

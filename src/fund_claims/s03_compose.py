"""Composition: compose categories from the flags by rule, and check the runs against each other.

The model never sees a category name. Everything here is deterministic:

  uses_ai + invests_ai_theme  ->  one of five mutually exclusive categories, with
                                  "both" allowed rather than a strict priority order
                                  that would drop it
  the four test_* flags       ->  a single ordinal "strictest bar stated", because a
                                  fund may state more than one and the falsifiability
                                  question is about the strongest commitment it makes
  the five chain_* flags      ->  kept multi-label, plus a count and a "names any
                                  position at all" indicator

It also runs two checks the model could fail, which are worth reporting rather than
silently repairing:

  * chain or test flags set while invests_ai_theme is N
  * risk_only set at the same time as uses_ai or invests_ai_theme
"""
import collections, io, json, os
import pandas as pd

from src import paths as tp  # noqa: E402

OUT = tp.out('fund_claims')
LOGS = tp.logs('fund_claims')
os.makedirs(OUT, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)

CHAIN = ['chain_developer', 'chain_enabler', 'chain_adopter',
         'chain_benefit_vague', 'chain_unspecified']
# "no test stated" is NOT asked of the model any more. As a residual question it was
# by far the least reliable thing in the schema - agreement 89.1%, kappa 0.496, against
# 0.87-0.91 for the three positive tests - so it is derived here instead.
TESTS = ['test_revenue', 'test_index_provider', 'test_manager']
SCOPE = ['ai_selection_criterion', 'ai_only_in_holding_description']
OWN = ['uses_ai', 'ai_operator_third_party', 'uses_ai_hedged', 'uses_ai_support']
FLAGS = OWN + ['invests_ai_theme'] + SCOPE + CHAIN + TESTS + ['risk_only']
# most falsifiable first: a number on the company's own books, then a named third
# party's public methodology, then the adviser's opinion, then nothing at all
TEST_ORDER = ['test_revenue', 'test_index_provider', 'test_manager']
TEST_NAME = {'test_revenue': 'revenue threshold',
             'test_index_provider': 'index provider',
             'test_manager': 'manager judgement'}


def evidence_meta():
    """(accession, series) -> (series_resolved, sha256 of the CURRENT evidence block)."""
    import gzip
    meta = {}
    p = os.path.join(OUT, 'evidence.jsonl.gz')
    if not os.path.exists(p):
        return meta
    with gzip.open(p, 'rt', encoding='utf-8') as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            meta[(r['accession'], r.get('series_id') or '')] = (
                bool(r.get('series_resolved', bool(r.get('series_id')))),
                r.get('evidence_sha256'))
    return meta


def load(run):
    rows = []
    p = os.path.join(OUT, 'labels_run%s.jsonl' % run)
    if not os.path.exists(p):
        return pd.DataFrame()
    meta = evidence_meta()
    seen = set()
    for line in io.open(p, encoding='utf-8'):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get('error'):
            continue
        # only a verdict on the text the evidence file holds NOW; a label for text that
        # has since moved to another fund, or been cut to a remainder, is history
        cur = meta.get((r['accession'], r.get('series_id') or ''))
        if cur is None or cur[1] != r.get('evidence_sha256'):
            continue
        k = (r['accession'], r['series_id'])
        if k in seen:
            continue
        seen.add(k)
        rows.append(r)
    D = pd.DataFrame(rows)
    for f in FLAGS:
        D[f] = D[f].eq('Y')
    # A null series_id would be dropped by any merge (NaN never matches NaN), which is
    # exactly how 123 filings vanished before. Give each unresolved filing its own
    # surrogate key and carry a flag saying so.
    D['series_resolved'] = [meta.get((a, s or ''), (bool(s), None))[0]
                            for a, s in zip(D.accession, D.series_id)]
    D['series_id'] = [s if s else '__UNRESOLVED__' + str(a)
                      for a, s in zip(D.accession, D.series_id)]
    return D


def compose(D):
    D = D.copy()
    # What the fund claims about its OWN process, as an ordered strength. Keeping
    # hedged and support use as their own levels matters: folding the strict uses_ai
    # test straight into "No AI" would make Goldman Sachs' boilerplate - "the Investment
    # Adviser MAY utilize artificial intelligence techniques ... to help extract
    # information from textual or audio datasets" - identical to a prospectus that never
    # mentions AI. Those are three different statements.
    D['own_claim'] = 'none'
    D.loc[D.uses_ai_support, 'own_claim'] = 'support'
    D.loc[D.uses_ai_hedged & ~D.uses_ai, 'own_claim'] = 'hedged'
    D.loc[D.uses_ai, 'own_claim'] = 'affirmative'

    D['category'] = 'No AI'
    D.loc[D.risk_only & (D.own_claim == 'none') & ~D.invests_ai_theme,
          'category'] = 'AI Risk Disclosure'
    D.loc[(D.own_claim == 'support') & ~D.invests_ai_theme, 'category'] = 'AI Support only'
    D.loc[(D.own_claim == 'hedged') & ~D.invests_ai_theme, 'category'] = 'Uses AI (hedged)'
    D.loc[D.invests_ai_theme & ~D.uses_ai, 'category'] = 'AI Theme'
    D.loc[D.uses_ai & ~D.invests_ai_theme, 'category'] = 'Uses AI'
    D.loc[D.uses_ai & D.invests_ai_theme, 'category'] = 'Uses AI + AI Theme'
    D['n_chain'] = D[CHAIN].sum(axis=1)
    D['chain_set'] = D[CHAIN].apply(
        lambda r: '|'.join(c.replace('chain_', '') for c in CHAIN if r[c]), axis=1)
    D['chain_specified'] = D[[c for c in CHAIN if c != 'chain_unspecified']].any(axis=1)

    def strictest(r):
        for t in TEST_ORDER:
            if r[t]:
                return TEST_NAME[t]
        return 'none stated'
    D['inclusion_test'] = D[TESTS].apply(strictest, axis=1)
    D['n_tests'] = D[TESTS].sum(axis=1)

    # Two readings of the theme claim, both kept. The broad one is every fund whose
    # holdings the text ties to AI at all; the strict one requires AI to be part of
    # the fund's own basis for choosing, rather than a fact about something it owns.
    # Recording both leaves the choice of population to the analysis rather than
    # burning it into the labels - the incidental group (single-stock and leveraged
    # products written on AI companies) is a usable sample in its own right.
    # The same move as on the theme side: uses_ai is now operator-agnostic and WHO
    # runs the AI is a separate field, because the two prompt wordings systematically
    # disagreed about AI that lives in an index methodology or a vendor tool - 27 of
    # the 36 fund-level disagreements, and stable within fund rather than random.
    D['own_inhouse'] = D.uses_ai & ~D.ai_operator_third_party
    D['own_thirdparty'] = D.uses_ai & D.ai_operator_third_party

    D['theme_broad'] = D.invests_ai_theme
    D['theme_strict'] = D.invests_ai_theme & D.ai_selection_criterion
    D['theme_incidental'] = (D.invests_ai_theme & D.ai_only_in_holding_description
                             & ~D.ai_selection_criterion)
    return D


def main():
    buf = io.StringIO()

    def w(s=''):
        buf.write(s + '\n')
        try:
            print(s)
        except UnicodeEncodeError:
            print(s.encode('ascii', 'replace').decode())

    runs = {}
    for run in ('1', '2'):
        D = load(run)
        if len(D):
            runs[run] = compose(D)

    w('=' * 96)
    w('COMPOSED LABELS')
    w('=' * 96)
    for run, D in runs.items():
        w('  run %s : %d filings, %d series' % (run, len(D), D.series_id.nunique()))
    w()

    for run, D in runs.items():
        w('-' * 96)
        w('RUN %s' % run)
        w('-' * 96)
        vc = D.category.value_counts()
        w('  category                  filings   series')
        for c in ['Uses AI', 'Uses AI + AI Theme', 'AI Theme', 'Uses AI (hedged)',
                  'AI Support only', 'AI Risk Disclosure', 'No AI']:
            sub = D[D.category == c]
            w('  %-24s %8d %8d' % (c, vc.get(c, 0), sub.series_id.nunique()))
        w()
        th = D[D.invests_ai_theme]
        if len(th):
            w('  value-chain position, among %d AI-theme filings:' % len(th))
            for c in CHAIN:
                w('    %-22s %6d  %5.1f%%'
                  % (c.replace('chain_', ''), th[c].sum(), 100 * th[c].mean()))
            w('    %-22s %6d  %5.1f%%' % ('names any position', th.chain_specified.sum(),
                                          100 * th.chain_specified.mean()))
            w()
            w('  most common combinations:')
            for k, n in collections.Counter(th.chain_set).most_common(10):
                w('    %5d  %s' % (n, k or '(none)'))
            w()
            w('  scope of the theme claim:')
            w('    %-22s %6d  %5.1f%%' % ('AI is a selection basis', th.theme_strict.sum(),
                                          100 * th.theme_strict.mean()))
            w('    %-22s %6d  %5.1f%%' % ('incidental only', th.theme_incidental.sum(),
                                          100 * th.theme_incidental.mean()))
            w()
            w('  strictest inclusion bar stated:')
            for k, n in th.inclusion_test.value_counts().items():
                w('    %-22s %6d  %5.1f%%' % (k, n, 100 * n / len(th)))
            w()

        w('  internal consistency (reported, not repaired):')
        bad1 = D[~D.invests_ai_theme & D[CHAIN].any(axis=1)]
        bad2 = D[~D.invests_ai_theme & D[TESTS].any(axis=1)]
        bad3 = D[D.risk_only & (D.uses_ai | D.invests_ai_theme)]
        bad4 = D[D.uses_ai & D.uses_ai_hedged]
        w('    chain flag set with invests_ai_theme = N : %d (%.1f%%)'
          % (len(bad1), 100 * len(bad1) / len(D)))
        w('    test flag set with invests_ai_theme = N  : %d (%.1f%%)'
          % (len(bad2), 100 * len(bad2) / len(D)))
        w('    risk_only set alongside a positive claim : %d (%.1f%%)'
          % (len(bad3), 100 * len(bad3) / len(D)))
        w('    uses_ai and uses_ai_hedged both set      : %d (%.1f%%)'
          % (len(bad4), 100 * len(bad4) / len(D)))
        w()
        w('  own-process claim, by strength:')
        for k, v in D.own_claim.value_counts().items():
            w('    %-14s %6d  %5.1f%%' % (k, v, 100 * v / len(D)))
        w()

    # ---------------- cross-run agreement ----------------
    if len(runs) == 2:
        A, B = runs['1'], runs['2']
        M = A.merge(B, on=['accession', 'series_id'], suffixes=('_1', '_2'))
        w('-' * 96)
        w('CROSS-RUN AGREEMENT  (same model, two differently worded prompts)')
        w('-' * 96)
        w('  filings classified by both : %d' % len(M))
        w()
        w('  %-22s %8s %9s %8s' % ('flag', 'agree', 'kappa', 'base Y'))
        for f in FLAGS:
            a, b = M[f + '_1'], M[f + '_2']
            po = (a == b).mean()
            pa, pb = a.mean(), b.mean()
            pe = pa * pb + (1 - pa) * (1 - pb)
            kap = (po - pe) / (1 - pe) if pe < 1 else float('nan')
            w('  %-22s %7.1f%% %9.3f %7.1f%%' % (f, 100 * po, kap, 100 * pa))
        w()
        agc = (M.category_1 == M.category_2).mean()
        w('  composed category agrees : %.1f%%' % (100 * agc))
        w('  inclusion test agrees    : %.1f%%'
          % (100 * (M.inclusion_test_1 == M.inclusion_test_2).mean()))
        w('  chain set agrees exactly : %.1f%%'
          % (100 * (M.chain_set_1 == M.chain_set_2).mean()))
        w()
        w('  where the composed category differs:')
        dif = collections.Counter('%-20s / %-20s' % (r.category_1, r.category_2)
                                  for r in M.itertuples()
                                  if r.category_1 != r.category_2)
        for k, n in dif.most_common(12):
            w('    %4d  %s' % (n, k))
        w()
        M.to_parquet(os.path.join(OUT, 'labels_both_runs.parquet'), index=False)
        w('wrote out/labels_both_runs.parquet')

    for run, D in runs.items():
        D.to_parquet(os.path.join(OUT, 'composed_run%s.parquet' % run), index=False)
        w('wrote out/composed_run%s.parquet' % run)

    with io.open(os.path.join(LOGS, 's03_compose.txt'), 'w', encoding='utf-8') as fh:
        fh.write(buf.getvalue())


if __name__ == '__main__':
    main()

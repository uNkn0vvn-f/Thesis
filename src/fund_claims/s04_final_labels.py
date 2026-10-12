"""The label file, with the taxonomy enforced and reliability reported.

Four rules:

1. ONE PRE-DECLARED POOLING RULE. The primary labels are RUN 1 ALONE: one clean
   measurement, no pooling artefact. Choosing, for each flag, whichever of {run 1, run 2,
   AND, OR} minimised attenuation would be arbitrary: re-estimating the error model under
   the other stratification flips the winner for 11 of the 12 flags it can be estimated
   for, and switching chain_unspecified from OR to AND moves 171 labels. A rule that
   depends on an unidentified nuisance choice cannot select the labels, so the three
   alternatives are reported as sensitivity with the number of labels each would change.

2. GRADES COMPUTED AFTER THE HIERARCHY IS ENFORCED, so that every entry of the grade
   table describes the variables actually saved.

3. FULL MUTUAL EXCLUSION, not only parent-child. Enforcing only "a child is false where
   its parent is false" leaves 60 filings both unspecified and specific, 14 both a
   selection criterion and holding-description-only, and 1 both risk-only and AI-support.

4. CROSS-RUN RELIABILITY, NOT IDENTIFICATION. kappa and lambda measure whether two
   promptings of one model agree. That is reliability, not identification: lambda is the
   OLS attenuation factor for a single binary regressor under non-differential
   misclassification in the SAME sample, and it does not transfer to a
   difference-in-differences, an event study, a fixed-effects panel or a fund-level
   `ever` treatment. No causal correction is offered here.

Filings whose series could not be resolved are carried with series_resolved = False and
excluded from fund-level aggregation BY NAME, with the count printed, rather than
vanishing inside a groupby.
"""
import io, os

import numpy as np
import pandas as pd

from src import paths as tp  # noqa: E402

OUT = tp.out('fund_claims')
os.makedirs(OUT, exist_ok=True)
LOGS = tp.logs('fund_claims')
os.makedirs(LOGS, exist_ok=True)
SRC = tp.out('fund_claims', 'labels_both_runs.parquet')
KAPPA_MIN = 0.80
PRIMARY_RULE = 'run 1 alone'

MODEL_FLAGS = ['uses_ai', 'ai_operator_third_party', 'uses_ai_hedged',
               'uses_ai_support', 'invests_ai_theme', 'ai_selection_criterion',
               'ai_only_in_holding_description', 'chain_developer', 'chain_enabler',
               'chain_adopter', 'chain_benefit_vague', 'chain_unspecified',
               'test_revenue', 'test_index_provider', 'test_manager', 'risk_only']
# `chain_benefit_vague` is deliberately not here: "a specific position was named" is
# exactly what it is not.
SPECIFIC_CHAIN = ['chain_developer', 'chain_enabler', 'chain_adopter']
TESTS = ['test_revenue', 'test_index_provider', 'test_manager']


def kappa(a, b):
    po = (a == b).mean()
    pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def pooled(a, b, rule):
    return {'run 1 alone': a, 'run 2 alone': b,
            'consensus (AND)': a & b, 'union (OR)': a | b}[rule]


def enforce(D):
    """Make the saved labels obey the schema the prompt states. Returns a report."""
    rep = []

    def drop(col, mask, why):
        n = int((D[col] & mask).sum())
        if n:
            D.loc[mask, col] = False
            rep.append((col, n, why))

    # 1. children require their parent
    for kid in (['ai_selection_criterion', 'ai_only_in_holding_description']
                + SPECIFIC_CHAIN + ['chain_unspecified'] + TESTS):
        drop(kid, ~D.invests_ai_theme, 'invests_ai_theme is false')
    drop('ai_operator_third_party', ~D.uses_ai, 'uses_ai is false')

    # 2. unspecified means NO specific role was named
    drop('chain_unspecified', D[SPECIFIC_CHAIN].any(axis=1),
         'a specific value-chain role is also set')

    # 3. "only in a holding description" cannot coexist with "is a selection basis"
    drop('ai_only_in_holding_description', D.ai_selection_criterion,
         'ai_selection_criterion is also set')

    # 4. risk-only means no positive claim of any kind
    drop('strategy_text_risk_only',
         D.uses_ai | D.uses_ai_hedged | D.uses_ai_support | D.invests_ai_theme,
         'a positive AI claim is also set')

    # 5. affirmative beats hedged
    drop('uses_ai_hedged', D.uses_ai, 'uses_ai is affirmative')
    return rep


def main():
    buf = io.StringIO()

    def w(s=''):
        buf.write(s + '\n')
        try:
            print(s)
        except UnicodeEncodeError:
            print(s.encode('ascii', 'replace').decode())

    M = pd.read_parquet(SRC)
    flags = [f for f in MODEL_FLAGS if f + '_1' in M.columns]

    out = M[['accession', 'series_id', 'series_name_1', 'date_1']].copy()
    out.columns = ['accession', 'series_id', 'series_name', 'date']
    if 'series_resolved_1' in M.columns:
        out['series_resolved'] = M.series_resolved_1.values
    else:
        out['series_resolved'] = out.series_id.notna()

    for f in flags:
        out[f] = pooled(M[f + '_1'], M[f + '_2'], PRIMARY_RULE).values
        out[f + '_agree'] = (M[f + '_1'] == M[f + '_2']).values
    # risk_only is scoped to the strategy text, not a full risk-factor section
    out = out.rename(columns={'risk_only': 'strategy_text_risk_only',
                              'risk_only_agree': 'strategy_text_risk_only_agree'})
    flags = ['strategy_text_risk_only' if f == 'risk_only' else f for f in flags]

    w('=' * 96)
    w('LABELS, TAXONOMY ENFORCED, RELIABILITY REPORTED')
    w('=' * 96)
    w('  filings                      : %d' % len(out))
    w('  primary pooling rule         : %s (pre-declared, not selected)' % PRIMARY_RULE)
    w('  series unresolved (excluded from fund level) : %d'
      % int((~out.series_resolved).sum()))
    w()

    # ---------------- enforcement, BEFORE grading ----------------
    w('-' * 96)
    w('1. TAXONOMY ENFORCEMENT (labels changed here, grades computed after)')
    w('-' * 96)
    rep = enforce(out)
    for col, n, why in rep:
        w('  %-32s %4d positives dropped: %s' % (col, n, why))
    if not rep:
        w('  nothing to drop')
    w()
    contradictions = {
        'unspecified AND a specific role':
            int((out.chain_unspecified & out[SPECIFIC_CHAIN].any(axis=1)).sum()),
        'selection basis AND holding-description-only':
            int((out.ai_selection_criterion
                 & out.ai_only_in_holding_description).sum()),
        'risk-only AND a positive claim':
            int((out.strategy_text_risk_only
                 & (out.uses_ai | out.invests_ai_theme | out.uses_ai_support)).sum()),
        'affirmative AND hedged':
            int((out.uses_ai & out.uses_ai_hedged).sum()),
        'child set with parent false':
            int((~out.invests_ai_theme & out[SPECIFIC_CHAIN + TESTS
                                             + ['chain_unspecified']].any(axis=1)).sum()),
    }
    w('  remaining contradictions, on the SAVED labels:')
    for k, v in contradictions.items():
        w('    %-46s %d' % (k, v))
    w()

    # ---------------- derived ----------------
    out['theme_broad'] = out.invests_ai_theme
    out['theme_strict'] = out.invests_ai_theme & out.ai_selection_criterion
    out['theme_incidental'] = (out.invests_ai_theme
                               & out.ai_only_in_holding_description)
    out['chain_specified'] = out[SPECIFIC_CHAIN].any(axis=1)
    out['own_inhouse'] = out.uses_ai & ~out.ai_operator_third_party
    out['own_thirdparty'] = out.uses_ai & out.ai_operator_third_party
    out['inclusion_test'] = 'none stated'
    for col, nm in reversed([('test_revenue', 'revenue threshold'),
                             ('test_index_provider', 'index provider'),
                             ('test_manager', 'manager judgement')]):
        out.loc[out[col], 'inclusion_test'] = nm
    out['falsifiable_test'] = out.test_revenue | out.test_index_provider

    # ---------------- grading, AFTER enforcement ----------------
    w('-' * 96)
    w('2. CROSS-RUN RELIABILITY  (this is reliability, NOT identification)')
    w('-' * 96)
    w('  %-32s %7s %8s %9s %7s' % ('flag', 'kappa', 'agree', 'grade', 'n=1'))
    rows = []
    for f in flags:
        src = 'risk_only' if f == 'strategy_text_risk_only' else f
        a, b = M[src + '_1'], M[src + '_2']
        k = kappa(a, b)
        grade = 'reliable' if k >= KAPPA_MIN else 'low-reliability'
        rows.append({'flag': f, 'kappa': k, 'cross_run_agree': float((a == b).mean()),
                     'grade': grade, 'rule': PRIMARY_RULE,
                     'n_positive_final': int(out[f].sum()),
                     'share_positive_final': float(out[f].mean())})
    G = pd.DataFrame(rows).sort_values(['grade', 'kappa'], ascending=[True, False])
    for _, r in G.iterrows():
        w('  %-32s %7.3f %7.1f%% %9s %7d'
          % (r['flag'], r['kappa'], 100 * r['cross_run_agree'], r['grade'],
             r['n_positive_final']))
    w()
    w('  kappa >= %.2f is labelled "reliable": the two promptings of one model agree' % KAPPA_MIN)
    w('  more than chance. It says nothing about whether they are RIGHT, and nothing')
    w('  about causal identification. No lambda correction is offered - the attenuation')
    w('  factor for a binary regressor does not transfer to DiD, event study, panel FE')
    w('  or a fund-level ever-treatment, and the error model it came from is itself')
    w('  sensitive to an arbitrary stratification choice.')
    w()

    # ---------------- what another rule would change ----------------
    w('-' * 96)
    w('3. SENSITIVITY: HOW MANY LABELS EACH ALTERNATIVE RULE WOULD MOVE')
    w('-' * 96)
    w('  %-32s %10s %10s %10s' % ('flag', 'run 2', 'AND', 'OR'))
    for f in flags:
        src = 'risk_only' if f == 'strategy_text_risk_only' else f
        a, b = M[src + '_1'], M[src + '_2']
        base = pooled(a, b, PRIMARY_RULE)
        w('  %-32s %10d %10d %10d'
          % (f, int((base != b).sum()), int((base != (a & b)).sum()),
             int((base != (a | b)).sum())))
    w()

    # ---------------- fund level ----------------
    w('-' * 96)
    w('4. FUND LEVEL  -- EVER-CLAIMED, WHICH IS NOT A TREATMENT VARIABLE')
    w('-' * 96)
    res = out[out.series_resolved & out.series_id.notna()]
    F = res.groupby('series_id').agg(
        n_filings=('accession', 'size'), series_name=('series_name', 'last'),
        first_date=('date', 'min'), last_date=('date', 'max'),
        **{v: (v, 'any') for v in ['uses_ai', 'theme_broad', 'theme_strict',
                                   'theme_incidental', 'chain_developer',
                                   'chain_enabler', 'chain_adopter',
                                   'chain_benefit_vague', 'chain_unspecified',
                                   'falsifiable_test', 'ai_operator_third_party',
                                   'uses_ai_support', 'uses_ai_hedged',
                                   'strategy_text_risk_only']})
    for v in ('uses_ai', 'theme_broad', 'theme_strict', 'theme_incidental'):
        w('  %-22s %4d funds' % (v, int(F[v].sum())))
    w('  funds total : %d  (from %d resolved filings; %d unresolved filings excluded)'
      % (len(F), len(res), int((~out.series_resolved).sum())))
    w()
    w('  WARNING, carried into the file itself. `any` pools 2020-2026, so a fund that')
    w('  first made a claim in 2025 is marked positive in 2020. These columns describe')
    w('  whether a fund EVER made a claim. They must not be used as a pre-period')
    w('  treatment for anything dated after the first filing they draw on; the as-of')
    w('  panel_fund_filing.parquet exists for that.')
    w()

    out.to_parquet(os.path.join(OUT, 'labels_final_filing.parquet'), index=False)
    F.to_parquet(os.path.join(OUT, 'labels_final_fund.parquet'))
    G.to_parquet(os.path.join(OUT, 'flag_grades.parquet'), index=False)
    w('wrote out/labels_final_filing.parquet (%d filings, %d cols)'
      % (len(out), out.shape[1]))
    w('wrote out/labels_final_fund.parquet   (%d funds, EVER-claimed)' % len(F))
    w('wrote out/flag_grades.parquet         (%d flags)' % len(G))

    with io.open(os.path.join(LOGS, 's04_final_labels.txt'), 'w',
                 encoding='utf-8') as fh:
        fh.write(buf.getvalue())


if __name__ == '__main__':
    main()

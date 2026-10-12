"""The thesis measurement (GPT-6 Luna labels) against the secondary one (DeepSeek labels).

Run after luna_measures.py and both passes of the label-dependent stages (run_all.py).

1. Claim prevalence by year under each set of fund labels (src.analysis.claims).
2. Fund flags on every classified filing, after taxonomy enforcement: counts and kappa.
3. Company roles on every Luna-labelled company-year: shares at >= 2, kappa, held value.
4. Role correspondence decomposed: fund claims from one model and company labels from the
   other, and DeepSeek's first company prompt alone (Luna answered only that one); each with
   the fund-cluster bootstrap and Holm adjustment of correspondence_inference.analyse.
5. Size-and-sector adjusted exposure under each pair of labels.
6. The claim ladder: rungs under the two label sets.
7. The event analysis's pre-event labels and the pre-event fund exposures built on them.

Tables: results/tables/robustness/cross_model_*.csv and cross_model_downstream.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.analysis import claims as CL  # noqa: E402
from src.analysis.robustness.common import (ROLES, TABLES_PRIMARY, PROCESSED, path, preevent_path,  # noqa: E402
                                            load_exposure_table, load_holdings, load_labels,
                                            load_buckets, load_claim_filings, build_exposure,
                                            attach_claims)
from src.analysis.robustness.correspondence_inference import analyse  # noqa: E402

OUT = TABLES_PRIMARY
FLAGS = ['theme_broad', 'theme_strict', 'uses_ai', 'uses_ai_hedged', 'uses_ai_support',
         'chain_developer', 'chain_enabler', 'chain_adopter', 'chain_specified',
         'chain_unspecified', 'chain_benefit_vague', 'falsifiable_test', 'strategy_text_risk_only']


def kappa(a, b):
    a, b = pd.Series(a).astype(bool), pd.Series(b).astype(bool)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def prevalence(log):
    den = pd.read_csv(PROCESSED / 'claim_denominator_fund_year.csv', dtype={'series_id': str})
    rows = []
    for lab in ('luna', 'deepseek'):
        T, _ = CL.annual_statistics(load_claim_filings(lab), den)
        A = T['claims_annual']
        for m in ('any_ai_mention', 'theme_broad', 'theme_strict', 'uses_ai', 'chain_specified',
                  'own_any', 'falsifiable_test'):
            col = m + '_rate_pct'
            for _, r in A.iterrows():
                rows.append({'labels': lab, 'year': int(r.year), 'measure': m,
                             'funds': int(r[m]), 'rate_pct': float(r[col]),
                             'denominator': int(r.funds_denominator)})
        V = T['claims_value_chain']
        for _, r in V.iterrows():
            rows.append({'labels': lab, 'year': int(r.year), 'measure': 'theme funds naming ' + r.position,
                         'funds': int(r.claiming_funds), 'rate_pct': float(r.share_pct),
                         'denominator': int(r.theme_funds)})
    P = pd.DataFrame(rows)
    W = P.pivot_table(index=['measure', 'year'], columns='labels', values=['funds', 'rate_pct']).reset_index()
    W.columns = ['_'.join(c).strip('_') for c in W.columns]
    W.to_csv(OUT / 'cross_model_prevalence.csv', index=False, float_format='%.4f')
    log['prevalence_2020_2025_pct'] = {
        m: {lab: [round(float(P[(P.labels == lab) & (P.measure == m) & (P.year == y)].rate_pct.iloc[0]), 2)
                  for y in (2020, 2025)] for lab in ('luna', 'deepseek')}
        for m in ('theme_broad', 'theme_strict', 'uses_ai')}


def fund_flags(log):
    A, B = load_claim_filings('luna'), load_claim_filings('deepseek')
    M = A.merge(B, on=['accession', 'series_id'], suffixes=('_luna', '_ds'), validate='one_to_one')
    rows = []
    for scope, S in (('all classified filings', M),
                     ('filings both label AI-themed', M[M.theme_broad_luna & M.theme_broad_ds])):
        for f in FLAGS:
            a, b = S[f + '_luna'].astype(bool), S[f + '_ds'].astype(bool)
            rows.append({'scope': scope, 'flag': f, 'filings': len(S), 'yes_luna': int(a.sum()),
                         'yes_deepseek': int(b.sum()), 'yes_both': int((a & b).sum()),
                         'agreement': float((a == b).mean()), 'kappa': kappa(a, b)})
    R = pd.DataFrame(rows)
    R.to_csv(OUT / 'cross_model_fund_flags.csv', index=False, float_format='%.4f')
    log['fund_flags_all_filings'] = R[R.scope == 'all classified filings'].set_index('flag')[
        ['yes_luna', 'yes_deepseek', 'kappa']].round(3).to_dict('index')


def deepseek_prompt_a():
    L = load_labels('deepseek')
    for r in ROLES:
        L[r] = L[r + '_1']
    return L


def company_roles(log, H, Ll, Ld, Ls):
    k = ['gvkey', 'yr']
    M = Ll[k + list(ROLES) + ['has_evidence']].merge(Ld[k + list(ROLES) + [r + s for r in ROLES for s in ('_1', '_2')]],
                                                      on=k, suffixes=('_luna', '_ds'), validate='one_to_one')
    rows = []
    for r in ROLES:
        a, b, c, d = (M[r + '_luna'] >= 2), (M[r + '_ds'] >= 2), (M[r + '_1'] >= 2), (M[r + '_2'] >= 2)
        rows.append({'role': r, 'company_years': len(M), 'share_luna': 100 * a.mean(),
                     'share_deepseek_min': 100 * b.mean(), 'share_deepseek_prompt_a': 100 * c.mean(),
                     'kappa_luna_vs_deepseek_min': kappa(a, b), 'kappa_luna_vs_deepseek_prompt_a': kappa(a, c),
                     'kappa_deepseek_a_vs_b': kappa(c, d)})
    R = pd.DataFrame(rows)
    # held value in the principal samples
    for lab, L in (('luna', Ll), ('deepseek', Ld)):
        E = load_exposure_table(lab)
        D = E[E.theme_broad & (E.labelled >= 50)][['series_id', 'q']]
        h = H.merge(D, on=['series_id', 'q']).merge(L[k + list(ROLES)], on=k)
        for i, r in enumerate(ROLES):
            R.loc[i, 'value_share_in_own_sample_' + lab] = 100 * h.loc[h[r] >= 2, 'value'].sum() / h.value.sum()
    R.to_csv(OUT / 'cross_model_company_roles.csv', index=False, float_format='%.4f')
    log['company_roles'] = R.set_index('role').round(3).to_dict('index')
    # adopter share by year: Luna and DeepSeek on the same company-years, DeepSeek on all
    by = []
    for y, g in M.groupby('yr'):
        by.append({'year': int(y), 'company_years_luna': len(g),
                   'adopter_luna_pct': 100 * (g.adopter_luna >= 2).mean(),
                   'adopter_deepseek_same_pct': 100 * (g.adopter_ds >= 2).mean(),
                   'adopter_deepseek_universe_pct': 100 * (Ld.loc[Ld.yr == y, 'adopter'] >= 2).mean()})
    pd.DataFrame(by).to_csv(OUT / 'cross_model_adopter_by_year.csv', index=False, float_format='%.3f')
    log['adopter_by_year'] = pd.DataFrame(by).round(1).to_dict('records')
    # evidence: record share and held value share without filing or patent evidence
    E = load_exposure_table('luna')
    D = E[E.theme_broad & (E.labelled >= 50)][['series_id', 'q']]
    h = H.merge(D, on=['series_id', 'q']).merge(Ll[k + ['has_evidence']], on=k)
    log['luna_no_evidence'] = {'company_years': int((~Ll.has_evidence).sum()), 'of': len(Ll),
                               'record_share_pct': float(100 * (~Ll.has_evidence).mean()),
                               'value_share_in_sample_pct': float(100 * h.loc[~h.has_evidence, 'value'].sum() / h.value.sum())}


def correspondence(log, H, S, Ll, Ld, Ls):
    El, Ed = load_exposure_table('luna'), load_exposure_table('deepseek')
    LdA = deepseek_prompt_a()
    expo = {}

    def X(L):
        if id(L) not in expo:
            expo[id(L)] = build_exposure(H, L, buckets=S)
        return expo[id(L)]

    def sample(L, E):
        D = attach_claims(X(L), E)
        return D[D.theme_broad & (D.labelled >= 50)].copy()

    specs = [('A Luna claims, Luna labels (thesis measurement)', Ll, El, '{role}'),
             ('B DeepSeek claims, DeepSeek labels (secondary)', Ld, Ed, '{role}'),
             ('C Luna claims, DeepSeek labels', Ld, El, '{role}'),
             ('D DeepSeek claims, Luna labels', Ll, Ed, '{role}'),
             ('E Luna claims, DeepSeek first company prompt', LdA, El, '{role}'),
             ('F DeepSeek claims, DeepSeek first company prompt', LdA, Ed, '{role}'),
             ('H Luna claims, Luna labels, size-and-sector adjusted', Ll, El, 'abn_{role}'),
             ('J DeepSeek claims, DeepSeek labels, adjusted (full benchmark)', Ld, Ed, 'abn_{role}'),
             ('K Luna claims, DeepSeek labels, adjusted (full benchmark)', Ld, El, 'abn_{role}'),
             ('N DeepSeek claims, Luna labels, adjusted', Ll, Ed, 'abn_{role}')]
    res, samples = [], {}
    for name, L, E, out in specs:
        D = sample(L, E)
        samples[name[0]] = set(zip(D.series_id, D.q))
        R, _ = analyse(name, D, outcome=out)
        res.append(R)
        print(name, R[['role', 'avg_median_gap', 'ci_lo', 'ci_hi', 'p_boot_holm']].round(3).to_string(index=False), flush=True)
    # the fund-quarters common to the two principal samples, each measurement on them
    common = samples['A'] & samples['B']
    for name, L, E in (('L common fund-quarters, Luna claims and labels', Ll, El),
                       ('M common fund-quarters, DeepSeek claims and labels', Ld, Ed)):
        D = sample(L, E)
        D = D[[k in common for k in zip(D.series_id, D.q)]].copy()
        R, _ = analyse(name, D)
        res.append(R)
        print(name, R[['role', 'avg_median_gap', 'ci_lo', 'ci_hi', 'p_boot_holm']].round(3).to_string(index=False), flush=True)
    R = pd.concat(res, ignore_index=True)
    R.to_csv(OUT / 'cross_model_correspondence.csv', index=False, float_format='%.6g')
    a, b = samples['A'], samples['B']
    fa, fb = {s for s, _ in a}, {s for s, _ in b}
    log['principal_samples'] = {'luna_fund_quarters': len(a), 'luna_funds': len(fa),
                                'deepseek_fund_quarters': len(b), 'deepseek_funds': len(fb),
                                'common_fund_quarters': len(a & b), 'common_funds': len(fa & fb),
                                'luna_only_funds': len(fa - fb), 'deepseek_only_funds': len(fb - fa)}
    # how often funds in each sample name each role (fund-quarters)
    for lab, L, E in (('luna', Ll, El), ('deepseek', Ld, Ed)):
        D = sample(L, E)
        log['principal_samples']['role_naming_pct_' + lab] = {
            r: float(100 * D['chain_' + r].mean()) for r in ROLES}
        log['principal_samples']['strict_theme_pct_' + lab] = float(
            100 * D.merge(E[['series_id', 'q', 'theme_strict']], on=['series_id', 'q']).theme_strict.astype(bool).mean())


def ladder(log):
    k = ['series_id', 'month']
    A = pd.read_parquet(path('ladder', 'luna'), columns=k + ['rung'])
    B = pd.read_parquet(path('ladder', 'deepseek'), columns=k + ['rung'])
    M = A.merge(B, on=k, suffixes=('_luna', '_ds'))
    T = pd.crosstab(M.rung_ds, M.rung_luna)
    T.to_csv(OUT / 'cross_model_ladder_rungs.csv')
    log['ladder'] = {'fund_months': len(M), 'same_rung_share': float((M.rung_luna == M.rung_ds).mean()),
                     'rung5_funds_luna': int(A.loc[A.rung == 5, 'series_id'].nunique()),
                     'rung5_funds_deepseek': int(B.loc[B.rung == 5, 'series_id'].nunique()),
                     'rung2_funds_luna': int(A.loc[A.rung == 2, 'series_id'].nunique()),
                     'rung2_funds_deepseek': int(B.loc[B.rung == 2, 'series_id'].nunique())}


def preevent(log):
    """The pre-event labels and the fund exposures built on them, model against model."""
    rows = []
    for tag in ('chatgpt', 'deepseek'):
        pl, pd_ = preevent_path(tag, 'label', 'luna'), preevent_path(tag, 'label', 'deepseek')
        if not pl.exists():
            continue
        A, B = pd.read_parquet(pl), pd.read_parquet(pd_)
        for X in (A, B):
            X['gvkey'] = X.gvkey.astype(str)
        M = A.merge(B[['gvkey', 'ai_rev'] + list(ROLES)], on='gvkey', suffixes=('_luna', '_ds'))
        rec = {'event': tag, 'companies': len(M),
               'ai_rev_corr': float(M[['ai_rev_luna', 'ai_rev_ds']].corr().iloc[0, 1]),
               'ai_rev_mean_luna': float(M.ai_rev_luna.mean()), 'ai_rev_mean_deepseek': float(M.ai_rev_ds.mean())}
        for r in ROLES:
            rec['kappa_' + r] = kappa(M[r + '_luna'] >= 2, M[r + '_ds'] >= 2)
        # fund level: the pre-event portfolio each event selects (filed before the event)
        cut = pd.Timestamp('2022-11-30' if tag == 'chatgpt' else '2025-01-27')
        ex = []
        for lab in ('luna', 'deepseek'):
            E = pd.read_parquet(preevent_path(tag, 'exposure', lab), columns=['series_id', 'report_date', 'filing_date', 'labelled', 'ai_rev'])
            E = E[(pd.to_datetime(E.filing_date) < cut) & (E.labelled >= 50)]
            ex.append(E.sort_values('report_date').groupby('series_id').tail(1).set_index('series_id').ai_rev.rename(lab))
        F = pd.concat(ex, axis=1).dropna()
        rec.update(funds=len(F), fund_exposure_corr=float(F.corr().iloc[0, 1]),
                   fund_exposure_sd_luna=float(F.luna.std()), fund_exposure_sd_deepseek=float(F.deepseek.std()))
        rows.append(rec)
    if rows:
        pd.DataFrame(rows).to_csv(OUT / 'cross_model_preevent.csv', index=False, float_format='%.4f')
        log['preevent'] = rows


def main():
    log = {}
    prevalence(log)
    fund_flags(log)
    H, S = load_holdings(), load_buckets()
    Ll, Ld = load_labels('luna'), load_labels('deepseek')
    company_roles(log, H, Ll, Ld, S)
    correspondence(log, H, S, Ll, Ld, S)
    ladder(log)
    preevent(log)
    (OUT / 'cross_model_downstream.json').write_text(json.dumps(log, indent=2, default=str), encoding='utf-8')
    print(json.dumps(log, indent=2, default=str))


if __name__ == '__main__':
    main()

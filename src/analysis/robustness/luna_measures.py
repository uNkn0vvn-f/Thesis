"""Luna labels as the thesis measurement, and the panels built on them.

The labels of OpenAI's GPT-6 Luna are the thesis measurement and the
DeepSeek labels are the secondary one. The choice was made on the models' public
capability (Luna has the higher intelligence index), after the 280-item agreement sample and
before any estimate on Luna labels was computed. Both sets are reported in full.

Luna was run once, with the first prompt wording on both sides (cross_model_labels.py):

  fund claims     Luna's answers to the fund prompt, with the taxonomy enforced as for
                  DeepSeek (src.fund_claims.s04_final_labels.enforce) and the same
                  derived variables (theme_broad, theme_strict, chain_specified, ...).
  company roles   Luna's 0-3 score on the first company prompt, counted at 2 or above
                  (the DeepSeek label is the lower of its two prompts' scores); AI revenue
                  share as Luna states it (-1, unknown, is missing). Whether filing text or
                  a patent profile was available is a property of the inputs and is taken
                  from the DeepSeek vintages, which were built from the same inputs.

Luna labels cover every AI-keyword fund filing and every company-year of the classified
universe (2,917 companies, annual vintages 2020-2026), as the DeepSeek vintages do, and the
3,006 companies of each pre-event information set (ChatGPT, DeepSeek), as the
DeepSeek pre-event labels do.

Writes
  data/processed/claim_filings_luna.csv         one row per classified fund filing
  data/processed/firm_label_vintages_luna.csv   one row per Luna-labelled company-year
  data/interim/cross_model/gpt-6-luna/derived/  panel_fund_filing.parquet,
      fund_quarter_exposure.parquet, build_log.json
  data/interim/analysis/flow_ladder_luna/     the claim ladder (claim_ladder.py) on Luna inputs
  data/interim/analysis/preevent_luna/        the event analysis's pre-event labels and exposures

Fed the DeepSeek labels, the panel, exposure and as-of routines here reproduce the
pipeline's tables; this is asserted on every run (and the ladder with --check-ladder).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src import paths as tp  # noqa: E402
from src.fund_claims.asof import expire  # noqa: E402
from src.fund_claims.s04_final_labels import enforce, SPECIFIC_CHAIN  # noqa: E402
from src.analysis.robustness.common import (PROCESSED, INTERIM, LUNA, LUNA_DERIVED, PATHS,  # noqa: E402
                                            PREEVENT_DIRS, preevent_path, load_holdings, load_buckets)

MODEL_FLAGS = ['uses_ai', 'ai_operator_third_party', 'uses_ai_hedged', 'uses_ai_support',
               'invests_ai_theme', 'ai_selection_criterion', 'ai_only_in_holding_description',
               'chain_developer', 'chain_enabler', 'chain_adopter', 'chain_benefit_vague',
               'chain_unspecified', 'test_revenue', 'test_index_provider', 'test_manager',
               'risk_only']
# the claim columns of the fund x filing panel (a01_build_panel.CLAIMS)
PANEL_CLAIMS = ['uses_ai', 'uses_ai_hedged', 'uses_ai_support', 'invests_ai_theme',
                'theme_broad', 'theme_strict', 'theme_incidental', 'chain_developer',
                'chain_enabler', 'chain_adopter', 'chain_benefit_vague',
                'chain_unspecified', 'falsifiable_test', 'strategy_text_risk_only']
# the claim columns of the fund-quarter exposure table (ai_exposure.CLAIMS)
EXPO_CLAIMS = ['uses_ai', 'theme_broad', 'theme_strict', 'theme_incidental',
               'chain_developer', 'chain_enabler', 'chain_adopter', 'chain_benefit_vague',
               'chain_unspecified', 'falsifiable_test']
POS = ('developer', 'enabler', 'adopter')
MEAS = ['ai_rev', 'core'] + list(POS)
LADDER_DIR = INTERIM / 'analysis' / 'flow_ladder_luna'


def read_jsonl(p: Path) -> pd.DataFrame:
    rows = []
    for line in open(p, encoding='utf-8'):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not r.get('error'):
            rows.append(r)
    return pd.DataFrame(rows).drop_duplicates('key', keep='last')


# ---------------------------------------------------------------- fund claims
def fund_labels(log: dict) -> pd.DataFrame:
    """Luna's fund labels in the layout of claim_filings.csv (s04_final_labels)."""
    C = pd.read_csv(PROCESSED / 'claim_filings.csv', dtype={'series_id': str})
    meta = C[['accession', 'series_id', 'series_name', 'date', 'series_resolved']].copy()
    R = read_jsonl(LUNA / 'fund.jsonl')
    # the evidence file leaves an unresolved series empty; the label files name it after the filing
    none = R.series_id.isna() | R.series_id.isin(['', 'None'])
    R.loc[none, 'series_id'] = '__UNRESOLVED__' + R.loc[none, 'accession']
    out = meta.merge(R[['accession', 'series_id'] + MODEL_FLAGS], on=['accession', 'series_id'],
                     how='left', validate='one_to_one')
    missing = out[MODEL_FLAGS[0]].isna()
    log['fund_filings'] = len(out)
    log['fund_filings_without_luna_label'] = int(missing.sum())
    if missing.any():
        raise SystemExit('%d classified filings have no Luna label; finish cross_model_labels.py run first'
                         % int(missing.sum()))
    for f in MODEL_FLAGS:
        out[f] = out[f].eq('Y')
    out = out.rename(columns={'risk_only': 'strategy_text_risk_only'})
    rep = enforce(out)
    log['fund_taxonomy_enforcement'] = [{'flag': c, 'dropped': n, 'reason': why} for c, n, why in rep]
    out['theme_broad'] = out.invests_ai_theme
    out['theme_strict'] = out.invests_ai_theme & out.ai_selection_criterion
    out['theme_incidental'] = out.invests_ai_theme & out.ai_only_in_holding_description
    out['chain_specified'] = out[SPECIFIC_CHAIN].any(axis=1)
    out['own_inhouse'] = out.uses_ai & ~out.ai_operator_third_party
    out['own_thirdparty'] = out.uses_ai & out.ai_operator_third_party
    out['inclusion_test'] = 'none stated'
    for col, nm in reversed([('test_revenue', 'revenue threshold'),
                             ('test_index_provider', 'index provider'),
                             ('test_manager', 'manager judgement')]):
        out.loc[out[col], 'inclusion_test'] = nm
    out['falsifiable_test'] = out.test_revenue | out.test_index_provider
    return out


def build_panel(P0: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """The fund x filing panel with claims from `labels` (as a01_build_panel merges them)."""
    assert not labels.duplicated(['accession', 'series_id']).any()
    keep = ['accession', 'series_id'] + [c for c in PANEL_CLAIMS if c in labels.columns]
    P = P0.drop(columns=[c for c in PANEL_CLAIMS if c in P0.columns])
    P = P.merge(labels[keep], on=['accession', 'series_id'], how='left', validate='many_to_one')
    for c in PANEL_CLAIMS:
        P[c] = P[c].astype('boolean').fillna(False).astype(bool) if c in P.columns else False
    return P[list(P0.columns)]


def harmonise_same_date(P: pd.DataFrame, log: dict) -> pd.DataFrame:
    """One claim per fund and prospectus date.

    The as-of designs need one claim per fund and date; the claim ladder takes the latest
    accession of the day and stops if two filings of that day disagree. Luna can answer the
    same evidence differently (no temperature setting), so where same-day filings disagree
    every row of the day takes the claims of its latest accession - the ladder's own rule -
    and the cases are logged."""
    P = P.copy()
    d = pd.to_datetime(P.date, errors='coerce')
    ok = P.series_resolved & P.series_id.notna() & d.notna()
    R = P[ok].assign(_d=d[ok])
    n = R.groupby(['series_id', '_d'])[PANEL_CLAIMS].nunique()
    bad = n[n.gt(1).any(axis=1)].index
    cases = []
    for sid, day in bad:
        rows = R[(R.series_id == sid) & (R._d == day)].sort_values('accession', kind='mergesort')
        last = rows.iloc[-1]
        cases.append({'series_id': sid, 'date': str(day.date()), 'accessions': rows.accession.tolist(),
                      'flags_differing': [c for c in PANEL_CLAIMS if rows[c].nunique() > 1],
                      'kept': last.accession})
        P.loc[rows.index, PANEL_CLAIMS] = last[PANEL_CLAIMS].to_numpy()
    log['same_date_conflicts_resolved'] = cases
    return P


def asof_claims(E0: pd.DataFrame, P: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Claims in force at each fund-quarter's report date (ai_exposure.py, same rule)."""
    P = P.copy()
    P['date'] = pd.to_datetime(P.date, errors='coerce')
    P = P[P.series_resolved & P.series_id.notna() & P.date.notna()]
    P = P[['series_id', 'date'] + [c for c in EXPO_CLAIMS if c in P.columns]].sort_values('date')
    base = E0.drop(columns=[c for c in EXPO_CLAIMS + ['date', 'claim_age_days'] if c in E0.columns])
    base = base.assign(report_date=pd.to_datetime(base.report_date)).sort_values('report_date')
    A = pd.merge_asof(base, P, left_on='report_date', right_on='date', by='series_id',
                      direction='backward')
    A, n = expire(A, 'report_date', [c for c in P.columns if c != 'series_id'])
    A['claim_age_days'] = (A.report_date - A.date).dt.days
    for c in EXPO_CLAIMS:
        A[c] = A[c].astype('boolean').fillna(False).astype(bool)
    return A, n


# ---------------------------------------------------------------- company labels
def company_labels(log: dict) -> pd.DataFrame:
    """Luna's company-year labels in the layout of firm_label_vintages.csv."""
    R = read_jsonl(LUNA / 'company.jsonl')
    R['gvkey'] = R.gvkey.astype(str).str.zfill(6)
    R['year'] = R.yr.astype(int)
    bad = R[list(POS)].isna().any(axis=1)
    log['company_years_luna'] = len(R)
    log['company_years_luna_unparsed_role'] = int(bad.sum())
    R = R[~bad].copy()
    for p in POS:
        R[p] = R[p].astype(int)
    s = pd.to_numeric(R.ai_sales_share, errors='coerce')
    R['ai_rev'] = s.where(s >= 0)
    R['ai_core'] = (R[['developer', 'enabler']].max(axis=1) >= 2) | (R.ai_rev >= 50)
    DS = pd.read_csv(PROCESSED / 'firm_label_vintages.csv', dtype={'gvkey': str})
    L = R[['year', 'gvkey']].merge(DS[['year', 'gvkey', 'has_text', 'has_pat', 'has_evidence']],
                                   on=['year', 'gvkey'], how='left', validate='one_to_one')
    log['company_years_luna_outside_deepseek_universe'] = int(L.has_evidence.isna().sum())
    L = L.merge(R[['year', 'gvkey', 'ai_core', 'developer', 'enabler', 'adopter', 'ai_rev']],
                on=['year', 'gvkey'])
    for c in ('has_text', 'has_pat', 'has_evidence'):
        L[c] = L[c].astype('boolean').fillna(False).astype(bool)
    return L.sort_values(['year', 'gvkey']).reset_index(drop=True)


def deepseek_company_labels() -> pd.DataFrame:
    """The DeepSeek vintages with ai_rev, read from the pipeline (mean of the two prompts)."""
    rows = []
    vint = INTERIM / 'firm_labels' / 'vintages' / '02_label'
    for d in sorted(p for p in vint.iterdir() if p.name.isdigit()):
        x = pd.read_parquet(d / 'firm_ai_labels.parquet', columns=['gvkey', 'ai_rev', 'ai_core'] + list(POS))
        x['gvkey'] = x.gvkey.astype(str)
        x['year'] = int(d.name)
        rows.append(x)
    return pd.concat(rows, ignore_index=True)


def exposures(H: pd.DataFrame, L: pd.DataFrame, S: pd.DataFrame) -> pd.DataFrame:
    """Fund-quarter measures on the labelled sleeve with size-decile x sector benchmarks
    (src/holdings/ai_exposure.py, same arithmetic)."""
    X = H.merge(L[['gvkey', 'year', 'ai_rev', 'ai_core'] + list(POS)].rename(columns={'year': 'yr'}),
                on=['gvkey', 'yr'], how='left')
    X['seen'] = X.ai_core.notna()
    X['m_core'] = X.ai_core.astype('boolean').fillna(False).astype(float) * 100
    for p in POS:
        X['m_' + p] = (X[p].fillna(-1) >= 2).astype(float) * 100
    X['m_ai_rev'] = X.ai_rev
    X = X.merge(S, on=['gvkey', 'q'], how='left')
    F = X[X.seen & X.bucket.notna()].drop_duplicates(['gvkey', 'q'])
    BM = (F.groupby(['q', 'bucket'])[['m_' + m for m in MEAS]].mean().reset_index()
          .rename(columns={'m_' + m: 'e_' + m for m in MEAS}))
    X = X.merge(BM, on=['q', 'bucket'], how='left')
    base = X.groupby(['series_id', 'q'], sort=False).agg(port_value=('value', 'sum')).reset_index()
    sub = X[X.seen]
    cov = sub.groupby(['series_id', 'q']).value.sum().rename('v_seen').reset_index()
    base = base.merge(cov, on=['series_id', 'q'], how='left')
    base['labelled'] = 100 * base.v_seen.fillna(0) / base.port_value
    for m in MEAS:
        col, ecol = 'm_' + m, 'e_' + m
        ok = sub[col].notna()
        a = (sub[ok].assign(_w=sub.loc[ok, 'value'] * sub.loc[ok, col])
             .groupby(['series_id', 'q']).agg(_w=('_w', 'sum'), _v=('value', 'sum')).reset_index())
        a[m] = a._w / a._v
        base = base.merge(a[['series_id', 'q', m]], on=['series_id', 'q'], how='left')
        eb = ok & sub[ecol].notna()
        b = (sub[eb].assign(_a=sub.loc[eb, 'value'] * sub.loc[eb, col],
                            _e=sub.loc[eb, 'value'] * sub.loc[eb, ecol])
             .groupby(['series_id', 'q']).agg(_a=('_a', 'sum'), _e=('_e', 'sum'), _v=('value', 'sum'))
             .reset_index())
        b['abn_' + m] = (b._a - b._e) / b._v
        base = base.merge(b[['series_id', 'q', 'abn_' + m]], on=['series_id', 'q'], how='left')
    return base.drop(columns='port_value')


LABEL_COLS = ['v_seen', 'labelled'] + [c for m in MEAS for c in (m, 'abn_' + m)]


def exposure_table(E0, H, L, S, P):
    """Pipeline columns that do not depend on a model label, plus label-based measures from L
    and as-of claims from panel P."""
    keep = [c for c in E0.columns if c not in LABEL_COLS + EXPO_CLAIMS + ['date', 'claim_age_days']]
    X = E0[keep].merge(exposures(H, L, S), on=['series_id', 'q'], how='left', validate='one_to_one')
    return asof_claims(X, P)


# ---------------------------------------------------------------- checks
def check_reproduction(E0, P0, H, S, log):
    """The routines reproduce the pipeline when fed the DeepSeek labels."""
    DS = pd.read_parquet(INTERIM / 'fund_claims' / 'labels_final_filing.parquet')
    P1 = build_panel(P0, DS)
    for c in PANEL_CLAIMS:
        assert (P1[c].to_numpy() == P0[c].astype(bool).to_numpy()).all(), 'panel differs: ' + c
    E1, _ = exposure_table(E0, H, deepseek_company_labels(), S, P0)
    M = E0.merge(E1, on=['series_id', 'q'], suffixes=('', '_re'), validate='one_to_one')
    diffs = {}
    for c in EXPO_CLAIMS:
        assert (M[c].astype(bool) == M[c + '_re']).all(), 'as-of claim differs: ' + c
    assert ((M.date == M.date_re) | (M.date.isna() & M.date_re.isna())).all(), 'claim date differs'
    for c in LABEL_COLS:
        a, b = M[c].astype(float), M[c + '_re'].astype(float)
        # summation order differs, so equality is relative (values up to 1e10 dollars)
        assert np.isclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True).all(), 'measure differs: ' + c
        diffs[c] = float(((a - b).abs() / a.abs().clip(lower=1)).max())
    log['reproduces_deepseek_pipeline'] = {'panel_rows': len(P0), 'fund_quarters': len(M),
                                           'max_relative_diff_measures': diffs}


EVENTS = ('chatgpt', 'deepseek')


def preevent_labels(log: dict) -> bool:
    """Luna's pre-event company labels, in the layout of the DeepSeek pre-event file.
    Returns False when the pre-event responses have not been collected."""
    p = LUNA / 'company_preevent.jsonl'
    if not p.exists():
        log['preevent'] = 'no Luna pre-event responses (cross_model_labels.py run --scope preevent)'
        return False
    R = read_jsonl(p)
    R['gvkey'] = R.gvkey.astype(str)
    for f in POS:
        R[f] = pd.to_numeric(R[f], errors='coerce')
    s = pd.to_numeric(R.ai_sales_share, errors='coerce')
    R['ai_rev'] = s.where(s >= 0)
    R['ai_core'] = (R[['developer', 'enabler']].max(axis=1) >= 2) | (R.ai_rev >= 50)
    info = {}
    for tag in EVENTS:
        D = pd.read_parquet(preevent_path(tag, 'label', 'deepseek'))
        D['gvkey'] = D.gvkey.astype(str)
        X = R[(R.event == tag) & R[list(POS)].notna().all(axis=1)]
        L = D[['gvkey', 'name', 'ctry', 'sic4', 'has_text', 'has_pat', 'has_evidence']].merge(
            X[['gvkey', 'ai_rev', 'ai_core'] + list(POS)], on='gvkey', how='inner', validate='one_to_one')
        for c in POS:
            L[c] = L[c].astype(int)
        dest = preevent_path(tag, 'label', 'luna')
        dest.parent.mkdir(parents=True, exist_ok=True)
        L.to_parquet(dest, index=False)
        info[tag] = {'companies_deepseek': len(D), 'companies_luna': len(L),
                     'role_ge2_pct_luna': {c: float(100 * (L[c] >= 2).mean()) for c in POS},
                     'ai_rev_mean_luna': float(L.ai_rev.mean()),
                     'ai_rev_mean_deepseek_same_firms': float(D.set_index('gvkey').ai_rev.reindex(L.gvkey).mean())}
    log['preevent_labels'] = info
    return True


def preevent_exposure(label: Path, panel: Path, out_dir: Path):
    """src.holdings.ai_exposure on one fixed label (the way the pipeline builds the
    pre-event exposures), with the claims taken from `panel`."""
    import os
    import subprocess
    code = ('import sys; sys.path.insert(0, sys.argv[1]); import src.holdings.ai_exposure as X; '
            'X.PANEL = sys.argv[2]; X.main()')
    env = dict(os.environ)
    env.update(LAB=str(label), EXP_OUT=str(out_dir), PYTHONIOENCODING='utf-8', THESIS_NO_API='1')
    r = subprocess.run([sys.executable, '-c', code, str(ROOT), str(panel)], env=env, cwd=ROOT,
                       capture_output=True, text=True, encoding='utf-8', errors='replace')
    if r.returncode:
        raise RuntimeError(r.stdout[-2000:] + r.stderr[-2000:])
    return out_dir / 'fund_quarter_exposure.parquet'


def build_preevent(log: dict):
    if not preevent_labels(log):
        return
    checks = {}
    for tag in EVENTS:
        # the routine reproduces the DeepSeek pre-event exposure from the DeepSeek inputs
        tmp = INTERIM / 'analysis' / 'preevent_check_deepseek' / tag
        e1 = pd.read_parquet(preevent_exposure(preevent_path(tag, 'label', 'deepseek'),
                                               PATHS['deepseek']['panel'], tmp))
        e0 = pd.read_parquet(preevent_path(tag, 'exposure', 'deepseek'))
        M = e0.merge(e1, on=['series_id', 'q'], suffixes=('', '_re'), validate='one_to_one')
        ok = len(M) == len(e0) == len(e1)
        for c in ('ai_rev', 'labelled', 'tech'):
            ok &= bool(np.isclose(M[c].astype(float), M[c + '_re'].astype(float), rtol=1e-9,
                                  atol=1e-9, equal_nan=True).all())
        for c in ('uses_ai', 'theme_broad'):
            ok &= bool((M[c].astype(bool) == M[c + '_re'].astype(bool)).all())
        assert ok, 'pre-event exposure build differs from the DeepSeek file for ' + tag
        out = PREEVENT_DIRS['luna'] / '03_exposure' / 'out' / tag
        E = pd.read_parquet(preevent_exposure(preevent_path(tag, 'label', 'luna'), PATHS['luna']['panel'], out))
        checks[tag] = {'reproduces_deepseek_preevent_exposure': True, 'fund_quarters': len(E),
                       'labelled_median_luna': float(E.labelled.median()),
                       'labelled_median_deepseek': float(e0.labelled.median())}
    log['preevent_exposure'] = checks


def run_ladder(panel: Path, exposure: Path, out_dir: Path):
    """The claim ladder script, pointed at other inputs and another output folder."""
    from src.analysis.flows import claim_ladder as L01
    L01.PANEL, L01.EXPO = str(panel), str(exposure)
    L01.OUT, L01.LOGS = str(out_dir / 'out'), str(out_dir / 'logs')
    (out_dir / 'out').mkdir(parents=True, exist_ok=True)
    (out_dir / 'logs').mkdir(parents=True, exist_ok=True)
    L01.main()
    return out_dir / 'out' / 'fund_month_ladder.parquet'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check-ladder', action='store_true',
                    help='also build the ladder on DeepSeek inputs and compare with the pipeline ladder')
    a = ap.parse_args(argv)
    log = {}
    LUNA_DERIVED.mkdir(parents=True, exist_ok=True)
    E0 = pd.read_parquet(PATHS['deepseek']['exposure'])
    P0 = pd.read_parquet(PATHS['deepseek']['panel'])
    H, S = load_holdings(), load_buckets()
    check_reproduction(E0, P0, H, S, log)

    F = fund_labels(log)
    drop = [c for c in F.columns if c.endswith('_agree')]
    F.drop(columns=drop).to_csv(PATHS['luna']['claim_filings'], index=False)
    L = company_labels(log)
    L.to_csv(PATHS['luna']['labels'], index=False)

    P = harmonise_same_date(build_panel(P0, F), log)
    for c in PANEL_CLAIMS:
        P[c] = P[c].astype(bool)
    P.to_parquet(PATHS['luna']['panel'], index=False)
    E, n_exp = exposure_table(E0, H, L, S, P)
    E.to_parquet(PATHS['luna']['exposure'], index=False)
    # the processed table read by run.py (fields as src/data/export_tables.py)
    fields = ['series_id', 'q', 'n_holdings', 'port_value', 'v_seen', 'report_date',
              'filing_date', 'date', 'claim_age_days', 'labelled', 'uses_ai', 'theme_broad',
              'chain_developer', 'chain_enabler', 'chain_adopter', 'developer', 'enabler', 'adopter']
    E[fields].sort_values(['series_id', 'q']).to_csv(PROCESSED / 'fund_quarter_exposure_luna.csv', index=False)
    log['exposure_fund_quarters'] = len(E)
    log['claims_expired_487_days'] = n_exp

    # coverage of the Luna company labels where the comparisons need them
    th = E[E.theme_broad]
    ds_cov = E0.set_index(['series_id', 'q']).labelled
    th = th.assign(labelled_deepseek=ds_cov.reindex(pd.MultiIndex.from_frame(th[['series_id', 'q']])).to_numpy())
    eligible = th.labelled_deepseek >= 50
    log['luna_theme_fund_quarters'] = len(th)
    log['luna_theme_fund_quarters_deepseek_coverage_ge50'] = int(eligible.sum())
    log['of_which_luna_coverage_ge50'] = int((eligible & (th.labelled >= 50)).sum())
    log['of_which_luna_coverage_shortfall_over_1pt'] = int((eligible & (th.labelled < th.labelled_deepseek - 1)).sum())

    if a.check_ladder:
        tmp = INTERIM / 'analysis' / 'flow_ladder_check_deepseek'
        lad = run_ladder(PATHS['deepseek']['panel'], PATHS['deepseek']['exposure'], tmp)
        A1, A0 = pd.read_parquet(lad), pd.read_parquet(PATHS['deepseek']['ladder'])
        same = (len(A1) == len(A0)) and (A1.rung.to_numpy() == A0.rung.to_numpy()).all()
        log['ladder_reproduces_pipeline'] = bool(same)
        assert same, 'ladder built on DeepSeek inputs differs from the pipeline ladder'
    run_ladder(PATHS['luna']['panel'], PATHS['luna']['exposure'], LADDER_DIR)
    build_preevent(log)
    (LUNA_DERIVED / 'build_log.json').write_text(json.dumps(log, indent=2, default=str), encoding='utf-8')
    print(json.dumps(log, indent=2, default=str))


if __name__ == '__main__':
    main()

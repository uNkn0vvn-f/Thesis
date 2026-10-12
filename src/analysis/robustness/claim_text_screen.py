"""Screen of AI-theme claim text: risk-factor boilerplate classified as a theme claim.

A rule-based screen, not a classifier replacement. A filing is flagged `risk_like` when
every sentence in its AI evidence that mentions AI is phrased as a risk or an adverse
possibility. Flags are candidates for manual checking; the sensitivity analyses drop them
to show how much the boilerplate matters, not to claim they are all errors.
"""
from __future__ import annotations

import gzip
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.analysis.robustness.common import (TABLES, INTERIM, PROCESSED, LABELS,  # noqa: E402
                                            load_claim_filings, load_claim_panel, load_exposure_table)

AI = re.compile(r'artificial intelligence|\bAI\b|machine learning|generative', re.I)
RISKY = re.compile(r'\brisks?\b|adversely|may (adversely )?impact|could result|challenges|'
                   r'\bharm\b|liabilit|uncertain|vulnerab|disrupt(ion|ive effects)', re.I)
SPLIT = re.compile(r'(?<=[.!?])\s+')
BOILER = re.compile(r'Artificial Intelligence Risk|collectively\W{1,3}AI Technologies', re.I)


def risk_like(text: str) -> bool:
    sents = [s for s in SPLIT.split(text or '') if AI.search(s)]
    return bool(sents) and all(RISKY.search(s) for s in sents)


def main():
    # Check provenance before writing any result: an empty glob must not imply no block attribution.
    from src.analysis.robustness.run_all import attributed_inputs, resolve
    attribution_files = [resolve(rel) for rel in attributed_inputs()]
    missing = [str(p) for p in attribution_files if not p.is_file()]
    if missing:
        raise FileNotFoundError('Missing attributed strategy shards:\n  ' + '\n  '.join(missing))
    rows = []
    with gzip.open(INTERIM / 'fund_claims' / 'evidence.jsonl.gz', 'rt', encoding='utf-8') as fh:
        for line in fh:
            r = json.loads(line)
            rows.append({'accession': r['accession'], 'series_id': r['series_id'],
                         'company': r['company'], 'evidence': r['evidence']})
    V = pd.DataFrame(rows)
    C = load_claim_filings()
    M = C.merge(V, on=['accession', 'series_id'], how='left')
    M['evidence_found'] = M.evidence.notna()
    M['risk_like'] = M.evidence.fillna('').map(risk_like)
    # boilerplate AI risk factor (e.g. "Artificial Intelligence Risk ... (collectively
    # "AI Technologies")") where AI is not the fund's selection basis
    M['ai_risk_boilerplate'] = (M.evidence.fillna('').str.contains(BOILER) & ~M.theme_strict)
    M['flag_boilerplate'] = M.theme_broad & (M.risk_like | M.ai_risk_boilerplate)
    M['year'] = pd.to_datetime(M.date).dt.year
    th = M[M.theme_broad]
    out = {'labels': LABELS, 'claim_filings': len(C), 'without_evidence_record': int((~M.evidence_found).sum()),
           'theme_filings': len(th), 'theme_filings_risk_like': int(th.risk_like.sum()),
           'theme_filings_ai_risk_boilerplate': int(th.ai_risk_boilerplate.sum()),
           'theme_filings_flag_boilerplate': int(th.flag_boilerplate.sum()),
           'flag_boilerplate_by_year': th.groupby('year').flag_boilerplate.sum().astype(int).to_dict(),
           'flag_boilerplate_funds': int(th[th.flag_boilerplate].series_id.nunique()),
           'theme_filings_risk_like_by_year': th.groupby('year').risk_like.sum().astype(int).to_dict(),
           'theme_filings_by_year': th.groupby('year').size().astype(int).to_dict(),
           'risk_like_theme_strict': int((th.risk_like & th.theme_strict).sum()),
           'risk_like_flagged_by_model_as_risk_only': int(th.loc[th.risk_like, 'strategy_text_risk_only'].sum()),
           'risk_like_top_registrants': th[th.risk_like].company.value_counts().head(10).to_dict()}
    # fund-years: annual prevalence uses `any` within the fund-year
    fy = M.groupby(['year', 'series_id']).agg(theme=('theme_broad', 'any'),
                                               theme_not_risk=('theme_broad', lambda s: bool((s & ~M.loc[s.index, 'flag_boilerplate']).any())))
    den = pd.read_csv(PROCESSED / 'claim_denominator_fund_year.csv', dtype={'series_id': str})
    nden = den.groupby('year').size()
    prev = pd.DataFrame({'theme_fund_years': fy.groupby(level=0).theme.sum(),
                         'theme_fund_years_excl_risk_like': fy.groupby(level=0).theme_not_risk.sum(),
                         'denominator': nden})
    prev['theme_pct'] = 100 * prev.theme_fund_years / prev.denominator
    prev['theme_pct_excl_risk_like'] = 100 * prev.theme_fund_years_excl_risk_like / prev.denominator
    prev.to_csv(TABLES / 'theme_prevalence_risk_screen.csv', float_format='%.4f')
    out['prevalence'] = prev.round(3).reset_index().to_dict('records')
    keep = ['accession', 'series_id', 'series_name', 'date', 'company', 'uses_ai', 'theme_broad',
            'theme_strict', 'chain_developer', 'chain_enabler', 'chain_adopter',
            'strategy_text_risk_only', 'evidence_found', 'risk_like', 'ai_risk_boilerplate',
            'flag_boilerplate']
    M[keep].to_csv(TABLES / 'claim_filings_risk_screen.csv', index=False)

    # attribution of the principal sample's theme filings: prospectus name against N-PORT name,
    # and whether the text came from single-series header recovery or block attribution
    E = load_exposure_table()[['series_id', 'labelled', 'theme_broad']]
    sample = set(E[E.theme_broad.astype('boolean').fillna(False).astype(bool) & (E.labelled >= 50)].series_id)
    P = load_claim_panel(columns=['accession', 'series_id', 'series_name', 'date', 'theme_broad'])
    P = P[P.series_id.isin(sample) & P.theme_broad.astype('boolean').fillna(False).astype(bool)].copy()
    H = pd.read_parquet(INTERIM / 'holdings' / 'holdings_bridged.parquet', columns=['series_id', 'series_name'])
    nport = H.drop_duplicates().groupby('series_id').series_name.agg(lambda x: ' | '.join(sorted(set(x))))
    P['nport_name'] = P.series_id.map(nport)
    norm = lambda x: set(re.sub(r'[^a-z0-9 ]', ' ', str(x).lower()).split())  # noqa: E731
    P['name_overlap_ok'] = [len(norm(a) & norm(b)) >= 2 or norm(a) <= norm(b) or norm(b) <= norm(a)
                            for a, b in zip(P.series_name, P.nport_name)]
    R = pd.read_csv(PROCESSED / 'claim_denominator_recovery.csv', dtype=str)
    P['header_recovered'] = [k in set(zip(R.accession, R.series_id)) for k in zip(P.accession, P.series_id)]
    blocks = set()
    for f in attribution_files:
        with gzip.open(f, 'rt', encoding='utf-8') as fh:
            for line in fh:
                r = json.loads(line)
                if r.get('attributed') == 'block':
                    blocks.add((r['accession'], r['series_id']))
    P['block_attributed'] = [k in blocks for k in zip(P.accession, P.series_id)]
    P.to_csv(TABLES / 'sample_theme_filings_attribution.csv', index=False)
    out['sample_theme_filings'] = len(P)
    out['sample_theme_filings_name_overlap_ok_share'] = float(P.name_overlap_ok.mean())
    out['sample_theme_filings_header_recovered'] = int(P.header_recovered.sum())
    out['sample_theme_filings_block_attributed'] = int(P.block_attributed.sum())
    (TABLES / 'claim_text_log.json').write_text(json.dumps(out, indent=2, default=str), encoding='utf-8')
    print(json.dumps(out, indent=2, default=str))


if __name__ == '__main__':
    main()

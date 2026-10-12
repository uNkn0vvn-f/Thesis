"""Merge the four layers into one frozen identification file, and hash it.

Everything downstream is supposed to read this and nothing else. So this script does
three jobs:

  1. ADJUDICATE the cases the name and claim-type steps leave open - the duplicate
     candidates and the borderline claim types - in a table that records the decision,
     the evidence and who made it, so the call is visible and reversible;
  2. MERGE the name spells (u02), the claim type (c01), the text labels (s03) and the
     label grades (s04) into fund-level, spell-level and filing-level files;
  3. FREEZE: write a manifest with the sha256 of every output and its row counts, so a
     later change to any of them is detectable.

The point-in-time structure survives the merge. A fund's AI name is a set of dated
spells, not a constant, and its text claims are dated filings. Nothing here collapses
either to a single value.
"""
import hashlib, io, json, os
import warnings
from datetime import datetime, timezone

import pandas as pd

pd.set_option('future.no_silent_downcasting', True)
warnings.filterwarnings('ignore', category=FutureWarning)

from src import paths as tp  # noqa: E402

ROOT = tp.code('src', 'fund_claims')
OUT = tp.out('fund_claims')
LOGS = tp.logs('fund_claims')
os.makedirs(OUT, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)

P_UNI = tp.out('fund_claims')
P_CLM = tp.out('fund_claims')
P_TXT = tp.out('fund_claims')
P_ERR = tp.out('fund_claims')

# --------------------------------------------------------------------------- #
# Adjudications. Each names the evidence that decided it; see the name-spell and claim-type logs.
# --------------------------------------------------------------------------- #
DUPLICATES = {
    # keep -> [absorbed]
    'S000084281': ['S000084279'],   # Alger AI Enablers & Adopters ETF absorbs the
                                    # mutual-fund twin: same strategy, two wrappers,
                                    # live concurrently for 761 days
    'S000092123': ['S000068068'],   # TrueShares absorbs TrueMark - one product moving
                                    # trusts, windows do not overlap (-41 days)
    'S000061746': ['S000061743'],   # Direxion: 61746 renamed Bull 3x -> Bull 2x in
                                    # Jan 2021 as 61743 stops filing; zero overlap, a
                                    # consolidation rather than two live products
}
DUP_REASON = {
    'S000084279': 'wrapper twin of S000084281 (ETF/mutual fund, concurrent)',
    'S000068068': 're-registration: TrueMark -> TrueShares (S000092123), sequential',
    'S000061743': 'Direxion consolidation into S000061746, zero overlap',
}
# Borderline claim types from c01_claim_type. Blank means "the rule stands"; a value overrides it.
CLAIM_OVERRIDE = {}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    buf = io.StringIO()

    def w(s=''):
        buf.write(s + '\n')
        try:
            print(s)
        except UnicodeEncodeError:
            print(s.encode('ascii', 'replace').decode())

    uni = pd.read_parquet(os.path.join(P_UNI, 'ai_universe.parquet'))
    spells = pd.read_parquet(os.path.join(P_UNI, 'ai_name_spells.parquet'))
    claim_fund = pd.read_parquet(os.path.join(P_CLM, 'claim_by_fund.parquet'))
    claim_spell = pd.read_parquet(os.path.join(P_CLM, 'claim_by_spell.parquet'))
    txt_fil = pd.read_parquet(os.path.join(P_ERR, 'labels_final_filing.parquet'))
    txt_fund = pd.read_parquet(os.path.join(P_ERR, 'labels_final_fund.parquet'))
    grades = pd.read_parquet(os.path.join(P_ERR, 'flag_grades.parquet'))

    w('=' * 96)
    w('FROZEN IDENTIFICATION LAYER')
    w('=' * 96)
    w('  inputs')
    w('    01 universe     %4d ever-AI-named series, %d name spells' % (len(uni), len(spells)))
    w('    02 claim type   %4d funds classified theme/process/overlay' % len(claim_fund))
    w('    03+04 text      %4d filings, %d series, %d graded flags'
      % (len(txt_fil), txt_fil.series_id.nunique(), len(grades)))
    w()

    # ---------------- 1. adjudication ----------------
    w('-' * 96)
    w('1. ADJUDICATIONS')
    w('-' * 96)
    absorbed = {d: k for k, v in DUPLICATES.items() for d in v}
    w('  duplicate series collapsed: %d' % len(absorbed))
    for d, k in absorbed.items():
        nm_d = uni.ai_name.get(d, '?')
        nm_k = uni.ai_name.get(k, '?')
        w('    %s  %-46s' % (d, str(nm_d)[:46]))
        w('       -> into %s  %-40s' % (k, str(nm_k)[:40]))
        w('       %s' % DUP_REASON[d])
    w()
    bl = pd.read_parquet(os.path.join(P_CLM, 'claim_borderline.parquet'))
    w('  borderline claim types carried forward WITHOUT override: %d' % len(bl))
    w('    (the rule stands; each is listed in claim_borderline.parquet with')
    w('     its reason, and CLAIM_OVERRIDE here is the single place to change one)')
    w()

    # ---------------- 2. fund-level merge ----------------
    F = uni.copy()
    F['ai_named'] = True
    F = F.join(claim_fund[['claim_last', 'claims']], how='left')
    F = F.rename(columns={'claim_last': 'name_claim_type'})

    T = txt_fund.copy()
    # NOT "text_claim". Every series here passed the AI-KEYWORD screen; whether it
    # makes a claim is a classification result, not a screening result. Calling the
    # screen a claim inflated the count from 722 to 923.
    T['ai_keyword_candidate'] = True
    T['makes_ai_claim'] = (T.uses_ai.fillna(False).astype(bool)
                           | T.theme_broad.fillna(False).astype(bool))
    keep = ['uses_ai', 'theme_broad', 'theme_strict', 'theme_incidental',
            'chain_developer', 'chain_enabler', 'chain_adopter', 'chain_benefit_vague',
            'chain_unspecified', 'falsifiable_test', 'ai_operator_third_party',
            'n_filings', 'first_date', 'last_date', 'series_name']
    T = T[[c for c in keep if c in T.columns]
            + ['ai_keyword_candidate', 'makes_ai_claim']]
    T.columns = ['text_' + c if c in ('series_name', 'n_filings', 'first_date',
                                      'last_date') else c for c in T.columns]

    ALL = F.join(T, how='outer')
    ALL['ai_named'] = ALL.ai_named.fillna(False).astype(bool)
    ALL['ai_keyword_candidate'] = ALL.ai_keyword_candidate.fillna(False).astype(bool)
    ALL['makes_ai_claim'] = ALL.makes_ai_claim.fillna(False).astype(bool)
    ALL['absorbed_into'] = pd.Series(absorbed)
    ALL['is_duplicate'] = ALL.index.isin(absorbed)
    # Previously the duplicates were only FLAGGED, so every downstream count -
    # including the adoption-trend figure - still counted both halves of each pair.
    # Collapse them for real: the absorbed row's claims are OR-ed into the survivor
    # and the absorbed row is dropped from the analysis universe.
    BOOLS = [c for c in ALL.columns if ALL[c].dtype == bool
             and c not in ('is_duplicate', 'ai_named')]
    for dead, keep_id in absorbed.items():
        if dead in ALL.index and keep_id in ALL.index:
            for c in BOOLS:
                ALL.loc[keep_id, c] = bool(ALL.loc[keep_id, c]) or bool(ALL.loc[dead, c])
    ALL['in_analysis_universe'] = ~ALL.is_duplicate
    ALL['display_name'] = ALL.ai_name.fillna(ALL.get('text_series_name'))

    w('-' * 96)
    w('2. THE MERGED UNIVERSE')
    w('-' * 96)
    w('  %-42s %6s' % ('', 'series'))
    w('  %-42s %6d' % ('AI name only', int((ALL.ai_named & ~ALL.ai_keyword_candidate).sum())))
    w('  %-42s %6d' % ('AI-keyword candidate only', int((~ALL.ai_named & ALL.ai_keyword_candidate).sum())))
    w('  %-42s %6d' % ('both', int((ALL.ai_named & ALL.ai_keyword_candidate).sum())))
    w('  %-42s %6d' % ('TOTAL in the identification layer', len(ALL)))
    w('  %-42s %6d' % ('of which flagged duplicates', int(ALL.is_duplicate.sum())))
    w()
    both = ALL[ALL.ai_named & ALL.ai_keyword_candidate]
    w('  DO THE TWO INSTRUMENTS AGREE, on the %d funds both cover?' % len(both))
    if len(both):
        pt = pd.crosstab(both.name_claim_type,
                         both.theme_broad.fillna(False).astype(bool).map({True: 'text: theme',
                                                             False: 'text: no theme'}))
        w(pt.to_string())
        w()
        proc = both[both.name_claim_type == 'process']
        w('  of %d PROCESS-named funds, %d also state in the prospectus that AI helps'
          % (len(proc), int(proc.uses_ai.fillna(False).astype(bool).sum())))
        w('  decide their holdings (%.0f%%)'
          % (100 * proc.uses_ai.fillna(False).astype(bool).mean() if len(proc) else 0))
        thm = both[both.name_claim_type == 'theme']
        w('  of %d THEME-named funds, %d state an AI theme in the prospectus (%.0f%%)'
          % (len(thm), int(thm.theme_broad.fillna(False).astype(bool).sum()),
             100 * thm.theme_broad.fillna(False).astype(bool).mean() if len(thm) else 0))
    w()

    # ---------------- 3. spell level ----------------
    S = claim_spell[['series_id', 'series_name', 'claim', 'date_from', 'date_to',
                     'registrant_name']].copy()
    S = S.rename(columns={'claim': 'name_claim_type'})
    S['is_duplicate'] = S.series_id.isin(absorbed)

    # ---------------- 4. write and hash ----------------
    files = {
        'identification_fund.parquet': ALL,
        'identification_spell.parquet': S,
        'identification_filing.parquet': txt_fil,
        'flag_grades.parquet': grades,
    }
    for nm, df in files.items():
        df.to_parquet(os.path.join(OUT, nm))
    adj = pd.DataFrame([{'series_id': d, 'absorbed_into': k, 'reason': DUP_REASON[d]}
                        for d, k in absorbed.items()])
    adj.to_parquet(os.path.join(OUT, 'adjudications.parquet'), index=False)
    files['adjudications.parquet'] = adj

    # Provenance. Hashing only the outputs proves they were not edited after the
    # fact; it does not pin down what produced them. Hash the scripts, the prompts and
    # the raw inputs too, so a changed upstream file or a re-worded prompt is visible.
    import glob as _glob
    prov_files = {}
    for pat in (os.path.join(ROOT, '*.py'),):
        for f in sorted(_glob.glob(pat)):
            prov_files[os.path.relpath(f, ROOT).replace(os.sep, '/')] = sha256(f)
    raw_inputs = {}
    for f in (os.path.join(P_TXT, 'evidence.jsonl.gz'),
              os.path.join(P_TXT, 'labels_run1.jsonl'),
              os.path.join(P_TXT, 'labels_run2.jsonl'),
              os.path.join(P_UNI, 'name_panel.parquet')):
        if os.path.exists(f):
            raw_inputs[os.path.basename(f)] = sha256(f)
    prompts = {}
    try:
        import importlib
        _m = importlib.import_module('src.fund_claims.s02_classify')
        prompts = {'prompt_run1_sha256': hashlib.sha256(
                       _m.PROMPT.encode('utf-8')).hexdigest(),
                   'prompt_run2_sha256': hashlib.sha256(
                       _m.PROMPT_B.encode('utf-8')).hexdigest(),
                   'model': _m.MODEL, 'thinking': _m.THINKING}
    except Exception as exc:
        prompts = {'error': repr(exc)[:120]}

    manifest = {
        'frozen_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'scripts': prov_files,
        'raw_inputs': raw_inputs,
        'prompts': prompts,
        'files': {nm: {'sha256': sha256(os.path.join(OUT, nm)),
                       'rows': int(len(df)), 'cols': int(df.shape[1])}
                  for nm, df in files.items()},
        'counts': {
            'series_total': int(len(ALL)),
            'series_ai_named': int(ALL.ai_named.sum()),
            'series_ai_keyword_candidate': int(ALL.ai_keyword_candidate.sum()),
            'series_makes_ai_claim': int(ALL.makes_ai_claim.sum()),
            'series_both': int((ALL.ai_named & ALL.ai_keyword_candidate).sum()),
            'series_duplicate': int(ALL.is_duplicate.sum()),
            'name_theme': int((ALL.name_claim_type == 'theme').sum()),
            'name_process': int((ALL.name_claim_type == 'process').sum()),
            'name_overlay': int((ALL.name_claim_type == 'overlay').sum()),
            'text_uses_ai': int(ALL.uses_ai.fillna(False).astype(bool).sum()),
            'text_theme_broad': int(ALL.theme_broad.fillna(False).astype(bool).sum()),
            'text_theme_strict': int(ALL.theme_strict.fillna(False).astype(bool).sum()),
            'filings': int(len(txt_fil)),
            'flags_reliable': int((grades.grade == 'reliable').sum()),
            'flags_low_reliability':
                int((grades.grade == 'low-reliability').sum()),
        },
        'provenance': {
            'name_panel_sources': 'N-PORT FUND_REPORTED_INFO + 485BPOS series_name',
            'classifier_model': 'deepseek-flash, temperature 0, thinking disabled',
            'classifier_runs': 'two, same model, two differently worded prompts',
            'reliability': 'agreement between the two runs, per flag (s04_final_labels)',
        },
    }
    with io.open(os.path.join(OUT, 'FROZEN.json'), 'w', encoding='utf-8') as fh:
        json.dump(manifest, fh, indent=2)

    w('-' * 96)
    w('3. FROZEN')
    w('-' * 96)
    for nm, meta in manifest['files'].items():
        w('  %-34s %5d rows  %2d cols  %s' % (nm, meta['rows'], meta['cols'],
                                              meta['sha256'][:16]))
    w()
    for k, v in manifest['counts'].items():
        w('  %-34s %6d' % (k, v))
    w()
    w('wrote out/FROZEN.json with the hash and row count of every output')

    with io.open(os.path.join(LOGS, 'z01_freeze.txt'), 'w', encoding='utf-8') as fh:
        fh.write(buf.getvalue())


if __name__ == '__main__':
    main()

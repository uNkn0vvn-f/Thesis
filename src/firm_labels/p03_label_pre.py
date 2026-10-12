"""Combine two classifier runs into one firm label, by the pre-declared consensus rule.

    ordinal positions   the MINIMUM of the two runs  (the less generous reading)
    AI sales share      the MEAN of the two, -1 treated as missing
    ai_core             max(developer, enabler) >= 2  or  ai_rev >= 50
    has_ev_both/any     a quotable basis in both / either run

THREE POSITIONS. Developer, enabler and adopter; BENEFIT is the genus the three divide
rather than a fourth sibling (see the header of src/firm_labels/c01_classify.py).

EVIDENCE FLAGS. Whether the model knew a firm is not asked, because a self-report cannot
answer the objection it would exist for. The flags are facts about the INPUT: a firm whose
filing supports the sales figure produced a sentence that can be found in the text sent to
it, and a firm with no filing text cannot produce one at all. The quote is checked here, at
label time, against the same evidence file the classifier read.

    CLS_DIR    directory holding labels_ev_run3.jsonl and labels_ev_run4.jsonl
    LAB_OUT    directory for firm_ai_labels.parquet
    EVID       optional; the evidence directory, to verify sales_basis quotes
"""
import difflib, io, json, os, re, sys

import numpy as np
import pandas as pd

from src import paths as tp  # noqa: E402

LEVELS = ('developer', 'enabler', 'adopter')

CLS = os.environ['CLS_DIR']
OUTDIR = os.environ['LAB_OUT']
os.makedirs(OUTDIR, exist_ok=True)
UNI = tp.out('firm_labels', 'firm_universe.parquet')


def wkappa(a, b, k=4):
    """Quadratic-weighted kappa for ordinals on 0..k-1."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok].astype(int), b[ok].astype(int)
    if len(a) < 10:
        return np.nan, 0
    O = np.zeros((k, k))
    for i, j in zip(a, b):
        O[i, j] += 1
    W = np.array([[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)])
    ha = np.bincount(a, minlength=k) / len(a)
    hb = np.bincount(b, minlength=k) / len(b)
    E = np.outer(ha, hb) * len(a)
    den = (W * E).sum()
    return (1 - (W * O).sum() / den if den else np.nan), len(a)


def _norm(s):
    """Whitespace and typographic differences must not count as a failed quote."""
    s = str(s or '')
    for a, b in (('‘', "'"), ('’', "'"), ('“', '"'), ('”', '"'),
                 ('–', '-'), ('—', '-'), ('−', '-')):
        s = s.replace(a, b)
    s = re.sub(r'[  -​  　﻿]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip().lower()


def verify_quotes(B, w):
    """Check each run's `sales_basis` against the text the classifier was actually sent.

    This runs HERE rather than later because the evidence file is rewritten whenever the
    fetcher runs again, and a quote checked against a refetched file is not checked at all.
    The columns it writes are facts about the input - whether the firm had a filing, whether
    the sentence the model returned can be found in it - and they replace `recognised`,
    which was the model's own opinion of whether it knew the company.
    """
    EVID = os.environ.get('EVID')
    B['has_text'] = False
    B['has_pat'] = False
    for r in ('1', '2'):
        B['quote_ok_' + r] = pd.NA
    if not EVID or not os.path.isdir(EVID):
        w('  EVID not set; sales_basis quotes were NOT verified')
        B['has_evidence'] = False
        return

    BIZ = {}
    p = os.path.join(EVID, 'business_text.jsonl')
    if os.path.exists(p):
        with io.open(p, encoding='utf-8', errors='replace') as fh:
            for line in fh:
                try:
                    x = json.loads(line)
                except Exception:
                    continue
                if x.get('business_text'):
                    BIZ[str(x['gvkey'])] = _norm(x['business_text'])
    pat = set()
    pp = os.path.join(EVID, 'patent_profile.parquet')
    if os.path.exists(pp):
        pat = set(pd.read_parquet(pp, columns=['gvkey']).gvkey.astype(str))

    gv = B.gvkey.astype(str)
    B['has_text'] = gv.isin(set(BIZ))
    B['has_pat'] = gv.isin(pat)
    B['has_evidence'] = B.has_text | B.has_pat

    def one(g, q):
        q = _norm(q)
        if not q or q == 'none':
            return pd.NA                      # declined, not failed
        src = BIZ.get(str(g))
        if not src:
            return False                      # a quote where no text was sent
        if len(q) < 20 or q in src:
            return q in src
        best = 0.0
        for sent in re.split(r'(?<=[.;])\s+', src):
            if abs(len(sent) - len(q)) > max(len(q), 40):
                continue
            best = max(best, difflib.SequenceMatcher(None, q, sent).ratio())
            if best > 0.97:
                break
        return best >= 0.85

    w()
    w('  SALES_BASIS, checked against the text the classifier was sent')
    w('  %-8s %10s %10s %12s %14s' % ('run', 'quoted', 'verified', 'NOT FOUND', 'declined'))
    for r in ('1', '2'):
        col = 'sales_basis_' + r
        if col not in B.columns:
            continue
        B['quote_ok_' + r] = [one(g, q) for g, q in zip(gv, B[col])]
        v = B['quote_ok_' + r]
        quoted = int(v.notna().sum())
        ok = int((v == True).sum())                                  # noqa: E712
        w('  %-8s %10d %9d (%.0f%%) %11d %14d'
          % ('run ' + r, quoted, ok, 100 * ok / max(quoted, 1),
             int((v == False).sum()), int(v.isna().sum())))          # noqa: E712
    w('  A firm with no filing text cannot quote, so "declined" is the right answer there')
    w('  and marks a memory-only label as a fact about the input.')
    w('  firms with filing text %d | with a patent profile %d | with neither %d'
      % (int(B.has_text.sum()), int(B.has_pat.sum()), int((~B.has_evidence).sum())))
    w()


def load(run):
    rows = []
    with io.open(os.path.join(CLS, 'labels_ev_run%s.jsonl' % run), encoding='utf-8') as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return pd.DataFrame(rows).drop_duplicates('gvkey', keep='last')


def main():
    lines = []

    def w(s=''):
        lines.append(s)
        print(s)

    U = pd.read_parquet(UNI)
    U['gvkey'] = U.gvkey.astype(str)
    R1, R2 = load('3'), load('4')
    for d in (R1, R2):
        d['gvkey'] = d.gvkey.astype(str)
        if 'error' not in d.columns:
            d['error'] = None
    B = R1.merge(R2, on='gvkey', suffixes=('_1', '_2'))
    B = B[B.error_1.isna() & B.error_2.isna()]
    n_out = int((~B.gvkey.isin(set(U.gvkey))).sum())
    B = B.merge(U, on='gvkey', how='inner', suffixes=('', '_u'))
    w('  labelled firms no longer in the firm universe, left out: %d' % n_out)

    w('PRE-SHOCK FIRM LABEL  (%s)' % CLS)
    w('  run 3 %d firms, run 4 %d firms, both without error %d' % (len(R1), len(R2), len(B)))
    w('  %-14s %9s %9s %9s %6s' % ('position', 'w-kappa', 'exact %', 'within1 %', 'n'))
    for f in LEVELS:
        a, b = B[f + '_1'], B[f + '_2']
        kp, n = wkappa(a, b)
        ok = a.notna() & b.notna()
        w('  %-14s %9.3f %9.1f %9.1f %6d'
          % (f, kp, 100 * (a[ok] == b[ok]).mean(),
             100 * ((a[ok] - b[ok]).abs() <= 1).mean(), n))
    rv = B[['ai_sales_share_1', 'ai_sales_share_2']].replace(-1, np.nan).dropna()
    w('  AI sales share: r = %.3f, mean |diff| %.1fpp, n = %d'
      % (rv.iloc[:, 0].corr(rv.iloc[:, 1]), (rv.iloc[:, 0] - rv.iloc[:, 1]).abs().mean(),
         len(rv)))

    verify_quotes(B, w)
    for f in LEVELS:
        B[f] = B[[f + '_1', f + '_2']].min(axis=1)
    B['ai_sales'] = B[['ai_sales_share_1',
                       'ai_sales_share_2']].replace(-1, np.nan).mean(axis=1)
    # The prompt asks what share of revenue is EARNED BY SELLING into the chain; the
    # downstream scripts read that column as `ai_rev`, so `ai_rev` is kept as an identical
    # alias of `ai_sales`. Both hold the same numbers.
    B['ai_rev'] = B['ai_sales']
    B['ai_core'] = (B[['developer', 'enabler']].max(axis=1) >= 2) | (B.ai_rev >= 50)

    keep = ['gvkey', 'name', 'ctry', 'sic4', 'rank_claim', 'value_all', 'value_claim',
            'in_north_america', 'has_text', 'has_pat', 'has_evidence',
            'quote_ok_1', 'quote_ok_2',
            'ai_sales', 'ai_rev', 'ai_core'] + list(LEVELS) + \
           [f + s for f in LEVELS for s in ('_1', '_2')] + \
           ['ai_sales_share_1', 'ai_sales_share_2', 'business_1', 'business_2',
            'sales_basis_1', 'sales_basis_2']
    keep = [c for c in keep if c in B.columns]
    B[keep].to_parquet(os.path.join(OUTDIR, 'firm_ai_labels.parquet'), index=False)
    w('  ai_rev >= 50%%: %d firms (%.1f%% of held value); ai_core: %d'
      % (int((B.ai_rev >= 50).sum()),
         100 * B.loc[B.ai_rev >= 50, 'value_all'].sum() / B.value_all.sum(),
         int(B.ai_core.sum())))
    w('wrote %s (%d firms)' % (os.path.join(OUTDIR, 'firm_ai_labels.parquet'), len(B)))
    with io.open(os.path.join(OUTDIR, 'label.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()

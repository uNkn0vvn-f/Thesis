"""Fetch each firm's last 10-K filed in a window, for an information set closed at a cutoff.

A label built from each firm's most recent 10-K would weight earlier holdings by later firm
descriptions. In the event design that would let a firm that moved into AI because of
ChatGPT score high on a treatment meant to be fixed before it, so every information set,
an annual vintage or a pre-event set, reads only the filings in its own window.

Window is set by environment:

    LM_LO      earliest filing date to consider  (yyyymmdd)
    LM_HI      latest filing date to consider    (yyyymmdd)
    EVID_OUT   directory for business_text.jsonl

Everything else - the Item 1 extraction, the amendment exclusion, the streaming stop at
Item 1A, the resumable cache - comes from src/firm_labels/e01_fetch_business.py, so the
information sets differ only in their filing window.
"""
import io, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from . import e01_fetch_business as F                      # noqa: E402  (helpers only)

LO = int(os.environ['LM_LO'])
HI = int(os.environ['LM_HI'])
OUTDIR = os.environ['EVID_OUT']
os.makedirs(OUTDIR, exist_ok=True)
DEST = os.path.join(OUTDIR, 'business_text.jsonl')
WORKERS = int(os.environ.get('WORKERS', '4'))


def main():
    U = pd.read_parquet(F.UNI)
    U['gvkey'] = U.gvkey.astype(str)
    C = pd.read_parquet(F.COMP, columns=['gvkey', 'cik'])
    C['gvkey'] = C.gvkey.astype(str)
    C['cik'] = pd.to_numeric(C.cik, errors='coerce')
    lm = pd.read_csv(F.LM, usecols=['CIK', 'FILING_DATE', 'ACC_NUM', 'FORM_TYPE'])
    ft = lm.FORM_TYPE.astype(str).str.upper().str.strip()
    lm = lm[ft.isin(['10-K', '10-K405', '10-KSB', '10-KT'])]
    # The upper bound is what makes the label pre-determined; the lower bound only keeps
    # the fetch small. A firm with no 10-K inside [LO, HI] is better served by an older
    # one than by nothing at all - with nothing, the classifier answers from the name and
    # the industry code alone. FALLBACK_YEARS widens the search for those firms only.
    # Worth knowing before spending bandwidth on it: the fallback reaches 42 extra firms
    # (+0.1% of held value) at the ChatGPT cutoff and 134 (+2.3%) at the DeepSeek cutoff,
    # because the ~1,000 firms with no evidence mostly have no 10-K at all - they are
    # foreign issuers, not domestic filers whose filing fell outside a window.
    fb = int(os.environ.get('FALLBACK_YEARS', '0'))
    lm = lm[lm.FILING_DATE <= HI].sort_values('FILING_DATE')
    inwin = lm[lm.FILING_DATE >= LO].drop_duplicates('CIK', keep='last')
    if fb:
        lo2 = int(str(HI)[:4]) - fb
        lo2 = int('%d%s' % (lo2, str(HI)[4:]))
        older = lm[(lm.FILING_DATE >= lo2) & ~lm.CIK.isin(set(inwin.CIK))]
        older = older.drop_duplicates('CIK', keep='last')
        lm = pd.concat([inwin, older], ignore_index=True)
        print('fallback %d years: %d firms in window, %d recovered from earlier filings'
              % (fb, len(inwin), len(older)))
    else:
        lm = inwin
    M = (U.merge(C.drop_duplicates('gvkey'), on='gvkey', how='left')
         .merge(lm[['CIK', 'ACC_NUM', 'FILING_DATE']], left_on='cik', right_on='CIK',
                how='left'))
    M = M[M.ACC_NUM.notna() & M.cik.notna()].sort_values('value_claim', ascending=False)

    done = set()
    if os.path.exists(DEST):
        with io.open(DEST, encoding='utf-8') as fh:
            for line in fh:
                try:
                    x = json.loads(line)
                except Exception:
                    continue
                if x.get('business_text'):
                    done.add(x['gvkey'])
    todo = [{'gvkey': r.gvkey, 'name': r['name'], 'cik': r.cik, 'acc': r.ACC_NUM}
            for _, r in M.iterrows() if r.gvkey not in done]
    print('window %d..%d | firms with a 10-K in it: %d (%.1f%% of held value) | '
          'latest filing date used: %d'
          % (LO, HI, len(M), 100 * M.value_all.sum() / U.value_all.sum(),
             int(M.FILING_DATE.max())))
    assert int(M.FILING_DATE.max()) <= HI, 'a filing after the window slipped through'
    print('already fetched %d, to fetch %d' % (len(done), len(todo)))

    t0 = time.time()
    n = 0
    with io.open(DEST, 'a', encoding='utf-8') as fh, \
            ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for res in ex.map(F.work, todo):
            fh.write(json.dumps(res, ensure_ascii=False) + '\n')
            n += 1
            if n % 100 == 0:
                fh.flush()
                el = time.time() - t0
                print('  %5d / %5d  %.0f s  ok %d fail %d  %.1f GB'
                      % (n, len(todo), el, F.stat['ok'], F.stat['fail'],
                         F.stat['bytes'] / 1e9))
                sys.stdout.flush()
    print('done %d in %.0f s | ok %d fail %d | downloaded %.2f GB'
          % (n, time.time() - t0, F.stat['ok'], F.stat['fail'], F.stat['bytes'] / 1e9))
    # the accession list is written beside the text so the window can be checked later
    M[['gvkey', 'cik', 'ACC_NUM', 'FILING_DATE']].to_parquet(
        os.path.join(OUTDIR, 'accessions.parquet'), index=False)


if __name__ == '__main__':
    main()

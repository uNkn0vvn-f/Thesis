"""Index every annual report each universe firm has filed with the SEC, 10-K or not.

The 10-K fetch locates annual reports through the Loughran-McDonald index, a 10-X file with
10-K and 10-Q filings and their variants but no 20-F or 40-F. Foreign private issuers
(20-F) and Canadian MJDS filers (40-F), among them Taiwan Semiconductor, ASML and Alibaba,
would otherwise reach the classifier with no business description; p05_fetch_20f picks
their filings from this index.

EDGAR's submissions API returns each filer's whole filing history with form types,
filing dates and accession numbers, which is what a point-in-time pick needs. This script
caches that history once, so all information sets share one download.

"HAS A CIK" IS NOT "FILES AN ANNUAL REPORT". Tencent has a CIK and files no annual report -
its US listing is an unsponsored ADR. The cache records what each firm actually files,
including `none`, so the companies left without filing text are measured, not assumed.

RATE. The SEC asks for a declared User-Agent (src.paths.sec_user_agent) and no more than
ten requests a second, and returns 429 when that is exceeded. This runs one request at a
time with a fixed delay and backs off on 429.

    python -m src.firm_labels.p04_foreign_index
    FORCE=1 ...   re-query firms already cached
"""
import gzip
import io
import json
import os
import time
import urllib.error
import urllib.request

import pandas as pd

from src import paths as tp  # noqa: E402

OUT = tp.out('firm_labels')
os.makedirs(OUT, exist_ok=True)
DEST = os.path.join(OUT, 'annual_filing_index.parquet')

UNI = os.environ.get('UNIVERSE') or tp.out('firm_labels', 'firm_universe.parquet')
COMP = tp.data('wrds', 'comp_company.parquet')
UA = {'User-Agent': tp.sec_user_agent(),
      'Accept-Encoding': 'gzip, deflate'}
ANNUAL = ('10-K', '10-K405', '10-KSB', '10-KT', '20-F', '40-F')
DELAY = float(os.environ.get('EDGAR_DELAY', '0.13'))       # ~8 requests a second
FORCE = os.environ.get('FORCE', '0') == '1'


def get(url, tries=5):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=40) as r:
                raw = r.read()
            return gzip.decompress(raw) if raw[:2] == b'\x1f\x8b' else raw
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            if e.code == 429:
                time.sleep(1.5 * (k + 1))           # back off and try again
                continue
            if k == tries - 1:
                raise
            time.sleep(0.5 * (k + 1))
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(0.5 * (k + 1))
    return None


def history(cik):
    """Every annual filing this CIK has made: (form, filing date, accession)."""
    j = get('https://data.sec.gov/submissions/CIK%010d.json' % cik)
    if j is None:
        return None
    d = json.loads(j)
    out = []
    blocks = [d.get('filings', {}).get('recent', {})]
    # a long filer's older history sits in separate files, listed under `files`
    for f in d.get('filings', {}).get('files', [])[:4]:
        b = get('https://data.sec.gov/submissions/' + f['name'])
        time.sleep(DELAY)
        if b:
            blocks.append(json.loads(b))
    for rec in blocks:
        for form, date, acc in zip(rec.get('form', []), rec.get('filingDate', []),
                                   rec.get('accessionNumber', [])):
            if form in ANNUAL:
                out.append((form, date, acc))
    return out


def main():
    U = pd.read_parquet(UNI, columns=['gvkey', 'name', 'value_all'])
    U['gvkey'] = U.gvkey.astype(str)
    C = pd.read_parquet(COMP, columns=['gvkey', 'cik'])
    C['gvkey'] = C.gvkey.astype(str)
    U = U.merge(C.drop_duplicates('gvkey'), on='gvkey', how='left')
    U['cikn'] = pd.to_numeric(U.cik, errors='coerce')
    todo = U[U.cikn.notna()].drop_duplicates('cikn').copy()

    done = {}
    if os.path.exists(DEST) and not FORCE:
        old = pd.read_parquet(DEST)
        done = set(old.gvkey.astype(str))
        rows = old.to_dict('records')
        print('cached already: %d firms' % len(done))
    else:
        rows = []
    todo = todo[~todo.gvkey.isin(done)]
    print('firms with a CIK to query: %d (delay %.2fs)' % (len(todo), DELAY))

    t0, n, err = time.time(), 0, 0
    for _, r in todo.iterrows():
        n += 1
        try:
            h = history(int(r.cikn))
        except Exception as e:
            h, err = None, err + 1
            rows.append({'gvkey': r.gvkey, 'cik': int(r.cikn), 'form': 'ERROR',
                         'filing_date': None, 'accession': None, 'err': str(e)[:60]})
        else:
            if not h:
                rows.append({'gvkey': r.gvkey, 'cik': int(r.cikn), 'form': 'none',
                             'filing_date': None, 'accession': None, 'err': ''})
            for form, date, acc in (h or []):
                rows.append({'gvkey': r.gvkey, 'cik': int(r.cikn), 'form': form,
                             'filing_date': date, 'accession': acc, 'err': ''})
        time.sleep(DELAY)
        if n % 200 == 0:
            print('  %5d / %5d  %.0fs  errors %d' % (n, len(todo), time.time() - t0, err))
            pd.DataFrame(rows).to_parquet(DEST, index=False)

    R = pd.DataFrame(rows)
    R.to_parquet(DEST, index=False)
    print('done %d firms in %.0fs, %d errors' % (n, time.time() - t0, err))

    tot = U.value_all.sum()
    V = R.merge(U[['gvkey', 'value_all']].drop_duplicates('gvkey'), on='gvkey', how='left')
    kind = (V[V.form.isin(ANNUAL)].groupby('gvkey').form
            .apply(lambda s: '+'.join(sorted(set(s)))).rename('forms').reset_index()
            .merge(U[['gvkey', 'value_all']].drop_duplicates('gvkey'), on='gvkey'))
    print()
    print('%-28s %7s %13s' % ('annual forms on file', 'firms', 'held value %'))
    for f, g in kind.groupby('forms'):
        if len(g) >= 3 or g.value_all.sum() / tot > 0.002:
            print('%-28s %7d %12.2f%%' % (f[:28], len(g), 100 * g.value_all.sum() / tot))
    for_20f = kind[kind.forms.str.contains('20-F|40-F')]
    print()
    print('firms with a 20-F or 40-F on file : %d, %.2f%% of held value'
          % (len(for_20f), 100 * for_20f.value_all.sum() / tot))
    print('wrote %s' % os.path.relpath(DEST, tp.REPO))


if __name__ == '__main__':
    main()

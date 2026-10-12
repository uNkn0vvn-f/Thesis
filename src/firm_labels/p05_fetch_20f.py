"""Fetch the business section of each foreign issuer's last 20-F or 40-F before a cutoff.

Companies without 10-K text are concentrated among foreign private issuers, which file
20-F (Canadian issuers 40-F) instead, and the Loughran-McDonald 10-X index read by
p01_fetch_pre contains neither form. This module takes those filings from the annual
report index of p04_foreign_index.

WHAT IS EXTRACTED. A 20-F's analogue of Item 1 is **Item 4, "Information on the Company"**,
whose Item 4.B is a business overview. The section ends at Item 4A (Unresolved Staff
Comments) or Item 5 (Operating and Financial Review), whichever comes first. A 40-F is a
different animal - a wrapper that incorporates a Canadian Annual Information Form filed as
an exhibit - so for those the largest EXHIBIT is read rather than the wrapper.

POINT-IN-TIME. Only filings dated on or before `LM_HI` are
eligible and the latest of them is taken, so a 2022 vintage reads the 20-F that was public
at the end of 2021. The output is appended to the same `business_text.jsonl` in the same
shape, and the accession list is extended, so everything downstream - the classifier, the
as-of checks, `has_10k` - works unchanged.

    LM_HI      latest filing date to use   (yyyymmdd)
    EVID_OUT   the vintage's evidence directory
    FORMS      which forms to try, default "20-F,40-F"

Firms that already have a business description in that directory are skipped: a 10-K is
the better document and is never replaced.
"""
import html
import io
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from src import paths as tp  # noqa: E402

from . import e01_fetch_business as F                      # noqa: E402  (helpers only)

HI = int(os.environ['LM_HI'])
OUTDIR = os.environ['EVID_OUT']
FORMS = tuple(os.environ.get('FORMS', '20-F,40-F').split(','))
WORKERS = int(os.environ.get('WORKERS', '4'))
DEST = os.path.join(OUTDIR, 'business_text.jsonl')
IDX = tp.out('firm_labels', 'annual_filing_index.parquet')

# Item 4 ... (Item 4A | Item 5).  F.sp tolerates the whitespace that stripping inline HTML
# tags leaves inside a word, which is what made the 10-K extractor work at all. The header
# pattern runs to the end of "on the Company" so the section body starts at the body and
# not in the middle of the heading.
H4 = re.compile(F.sp('Item') + r'\s*4' + F.SEP + F.sp('Information')
                + r'\s*' + F.sp('on') + r'\s*' + F.sp('the') + r'\s*' + F.sp('Company'),
                re.I)
H4B = re.compile(F.sp('Item') + r'\s*4' + F.SEP + F.sp('Business'), re.I)
H_END = re.compile(F.sp('Item') + r'\s*(?:4\s*A' + F.SEP + F.sp('Unresolved')
                   + r'|5' + F.SEP + F.sp('Operating') + r')', re.I)
lock = threading.Lock()
stat = {'ok': 0, 'fail': 0, 'bytes': 0}


def norm(b):
    """Strip tags, then resolve the ENTITIES that hide a section header.

    Taiwan Semiconductor writes its headings as `ITEM&#8201;4. INFORMATION ON THE COMPANY`
    - a thin space between the word and the number. The shared `strip_html` resolves five
    entities by hand and leaves the rest as literal text, so `\\s*4` cannot match `&#8201;4`
    and the only "Item 4" the pattern could see was the table of contents, which uses plain
    spaces. Every extraction then came back as a cross-reference fragment. `html.unescape`
    resolves the whole entity table at once, and the Unicode space characters it produces -
    thin, hair, narrow no-break, zero-width - are folded to ordinary spaces so the existing
    whitespace-tolerant patterns work on them."""
    t = F.strip_html(b)
    t = html.unescape(t)
    t = re.sub(r'[   -​  　﻿]', ' ', t)
    return re.sub(r'\s+', ' ', t)


def best_item4(txt, bounded_only=True):
    """The longest Item 4 section that is CLOSED by an Item 4A or Item 5 header.

    Requiring the closing header is what makes this usable. A 20-F names Item 4 in three
    places - the table of contents, the section itself, and cross-references in the body
    ("see Item 4. Information on the Company - Capacity Management") - and a
    cross-reference has no closing header after it, so an unbounded match runs to the end
    of whatever has been downloaded and always looks like the longest one. Accepting those
    would stop the download inside the table of contents and return a sentence fragment
    for Taiwan Semiconductor. Only closed sections count; `bounded_only=False` exists for
    the final fallback, where nothing closed was found and a long body is better than none.
    """
    best, blen = None, -1
    for pat in (H4, H4B):
        for m in pat.finditer(txt):
            # A cross-reference is always quoted - `See "Item 4. Information on the
            # Company - Capacity Management"` - and a real heading never is. One character
            # of context separates the two cleanly in every 20-F inspected, and without it
            # the longest "section" is whatever follows a citation.
            if m.start() and txt[m.start() - 1] in '"“”':
                continue
            st = m.end()
            en = H_END.search(txt[st:])
            if en is None:
                if bounded_only:
                    continue
                body = txt[st:st + 20000]
            else:
                body = txt[st:st + en.start()]
            if len(body) > blen:
                best, blen = body, len(body)
    return best, blen


def biggest_doc(cik, acc, exhibits, rank=0):
    """The rank-th largest document in a filing. 40-F carries its content in an exhibit,
    so for those the exhibit filter is inverted rather than applied."""
    u = ('https://www.sec.gov/Archives/edgar/data/%d/%s/index.json'
         % (int(cik), acc.replace('-', '')))
    r = requests.get(u, headers=F.UA, timeout=60)
    if r.status_code != 200:
        return None
    items = json.loads(r.text)['directory']['item']
    cand = [it for it in items
            if it['name'].lower().endswith(('.htm', '.html', '.txt'))
            and 'index' not in it['name'].lower()
            and (bool(F.EXH.search(it['name'])) == exhibits)]
    if not cand:
        return None
    cand.sort(key=lambda it: int(it.get('size') or 0), reverse=True)
    if rank >= len(cand):
        return None
    return ('https://www.sec.gov/Archives/edgar/data/%d/%s/%s'
            % (int(cik), acc.replace('-', ''), cand[rank]['name']))


def section(cik, acc, form, rank=0):
    url = biggest_doc(cik, acc, exhibits=(form.startswith('40-F')), rank=rank)
    if not url:
        return None, 'no document'
    rr = requests.get(url, headers=F.UA, timeout=240, stream=True)
    if rr.status_code != 200:
        return None, 'doc HTTP %d' % rr.status_code
    buf = b''
    capped = True
    for chunk in rr.iter_content(262144):
        buf += chunk
        if len(buf) > F.MAXBYTES:
            break
        # only a CLOSED section may stop the download: an unbounded match inside the
        # table of contents would otherwise end it before the real section arrives
        if len(buf) % 1048576 < 262144 and best_item4(norm(buf))[1] >= 3000:
            capped = False
            break
    rr.close()
    with lock:
        stat['bytes'] += len(buf)
    txt = norm(buf)
    best, blen = best_item4(txt)
    if best is None or blen < 400:
        best, blen = best_item4(txt, bounded_only=False)
    if best is None:
        # a 40-F exhibit is often an AIF with no "Item 4" heading at all; its whole body
        # is the business description, so fall back to the head of the document
        if form.startswith('40-F') and len(txt) > 4000:
            return txt[:14000].strip(), 'ok 40-F body %d chars' % len(txt)
        return None, 'no Item 4 header in %d KB' % (len(buf) // 1024)
    if blen < 400:
        return None, 'Item 4 section too short (%d chars)' % blen
    return best.strip()[:14000], 'ok %d chars%s' % (blen, '' if not capped else ' (capped)')


def work(rec):
    body, note = None, 'not attempted'
    for rank in (0, 1, 2):
        try:
            body, note = section(rec['cik'], rec['acc'], rec['form'], rank)
        except Exception as e:
            body, note = None, repr(e)[:120]
        if body:
            if rank:
                note += ' (doc #%d)' % (rank + 1)
            break
    with lock:
        stat['ok' if body else 'fail'] += 1
    return {'gvkey': rec['gvkey'], 'name': rec['name'], 'cik': int(rec['cik']),
            'acc': rec['acc'], 'form': rec['form'], 'note': note,
            'business_text': body or ''}


def main():
    assert os.path.exists(IDX), ('build the annual filing index first: '
                                 'p04_foreign_index.py')
    U = pd.read_parquet(F.UNI)
    U['gvkey'] = U.gvkey.astype(str)
    I = pd.read_parquet(IDX)
    I['gvkey'] = I.gvkey.astype(str)
    I = I[I.form.isin(FORMS) & I.filing_date.notna()].copy()
    I['fd'] = I.filing_date.str.replace('-', '', regex=False).astype(int)
    I = I[I.fd <= HI].sort_values('fd')

    have = set()
    if os.path.exists(DEST):
        with io.open(DEST, encoding='utf-8') as fh:
            for line in fh:
                try:
                    x = json.loads(line)
                except Exception:
                    continue
                if x.get('business_text'):
                    have.add(str(x['gvkey']))

    pick = I[~I.gvkey.isin(have)].drop_duplicates('gvkey', keep='last')
    pick = pick.merge(U[['gvkey', 'name', 'value_all']], on='gvkey', how='inner')
    tot = U.value_all.sum()
    print('cutoff %d | firms already with text: %d' % (HI, len(have)))
    print('firms reachable by %s and still missing: %d (%.2f%% of held value)'
          % ('/'.join(FORMS), len(pick), 100 * pick.value_all.sum() / tot))
    # The fetch is skipped when there is nothing new, but the accession list below is
    # rebuilt on every run: p01_fetch_pre rewrites accessions.parquet from the 10-K index
    # whenever it runs, so the 20-F rows have to be added back after it.
    if len(pick):
        assert int(pick.fd.max()) <= HI, 'a filing after the cutoff slipped through'
        print('latest filing used: %d' % int(pick.fd.max()))

        todo = [{'gvkey': r.gvkey, 'name': r['name'], 'cik': r.cik, 'acc': r.accession,
                 'form': r.form} for _, r in pick.iterrows()]
        t0, n = time.time(), 0
        written = []
        with io.open(DEST, 'a', encoding='utf-8') as fh, \
                ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for res in ex.map(work, todo):
                fh.write(json.dumps(res, ensure_ascii=False) + '\n')
                written.append(res)
                n += 1
                if n % 50 == 0:
                    fh.flush()
                    print('  %4d / %4d  %.0fs  ok %d fail %d  %.2f GB'
                          % (n, len(todo), time.time() - t0, stat['ok'], stat['fail'],
                             stat['bytes'] / 1e9))
                    sys.stdout.flush()
        ok = [r for r in written if r['business_text']]
        got = U[U.gvkey.isin({r['gvkey'] for r in ok})].value_all.sum()
        print('done %d in %.0fs | ok %d fail %d | recovered %.2f%% of held value'
              % (n, time.time() - t0, stat['ok'], stat['fail'], 100 * got / tot))
    else:
        print('nothing new to fetch; updating the accession list only')

    # Extend the accession list so the as-of checks and `has_10k` see these filings. It is
    # built from the whole evidence file rather than from this run's new rows, because
    # p01_fetch_pre rewrites accessions.parquet from the 10-K index every time it runs.
    acc_p = os.path.join(OUTDIR, 'accessions.parquet')
    every = {}
    with io.open(DEST, encoding='utf-8') as fh:
        for line in fh:
            try:
                x = json.loads(line)
            except Exception:
                continue
            if x.get('form') and x.get('business_text') and x.get('acc'):
                every[str(x['gvkey'])] = (int(x['cik']), x['acc'])
    fd = I.drop_duplicates('gvkey', keep='last').set_index('gvkey').fd
    add = pd.DataFrame([{'gvkey': g, 'cik': c, 'ACC_NUM': a,
                         'FILING_DATE': int(fd[g])}
                        for g, (c, a) in every.items() if g in fd.index])
    if len(add):
        if os.path.exists(acc_p):
            A = pd.read_parquet(acc_p)
            A['gvkey'] = A.gvkey.astype(str)
            add = add[~add.gvkey.isin(set(A.gvkey))]
            A = pd.concat([A, add], ignore_index=True)
        else:
            A = add
        assert int(A.FILING_DATE.max()) <= HI, 'accession list now exceeds the cutoff'
        A.to_parquet(acc_p, index=False)
        print('accession list extended to %d rows' % len(A))


if __name__ == '__main__':
    main()

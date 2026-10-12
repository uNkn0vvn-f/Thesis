"""Item 1 (Business) of a 10-K, located exactly and streamed.

Compustat carries no business description, so without filing text the classifier would be
asked what it remembers about a company, cued by a name and an industry code. These helpers
read the company's own description of itself instead. Loughran-McDonald supplies the CIK
and accession number of every 10-K, so the document is located exactly; `work` returns the
Item 1 text of one filing, trying the primary document and then the next-largest ones.

A 10-K is 1-8 MB and Item 1 sits in the first fifth of it, so the document is streamed and
the download stops as soon as Item 1A (Risk Factors) has been seen. A fixed truncation would
return some companies' table of contents, because the real Item 1 lies past the cut.

Used by p01_fetch_pre (10-K) and p05_fetch_20f (the same streaming and HTML handling for
20-F and 40-F documents). Read-only public SEC data.
"""
import json, os, re, threading

import requests

from src import paths as tp  # noqa: E402

UNI = os.environ.get('UNIVERSE') or tp.out('firm_labels', 'firm_universe.parquet')
LM = tp.data('wrds', 'Loughran-McDonald_10X_Summaries_1993-2025.csv')
COMP = tp.data('wrds', 'comp_company.parquet')

UA = {'User-Agent': tp.sec_user_agent()}
MAXBYTES = int(os.environ.get('MAXBYTES', str(9_000_000)))
EXH = re.compile(r'(^|[/_-])(ex|exhibit)[-_0-9]', re.I)


def sp(word):
    """A pattern for `word` that tolerates whitespace between its letters.

    Stripping HTML tags turns an inline span inside a word into a space, so Teradyne's
    section header arrives as "Item 1: B usiness" and Beam's and Microsoft's arrive
    similarly broken. A literal "Business" then matches only the table of contents, and
    the extracted section is the three characters between two TOC lines. That is the
    "Item 1 section too short (3 chars)" failure - 127 firms carrying 13% of held value.
    """
    return r'\s*'.join(re.escape(c) for c in word)


SEP = r'\s*[\.\:\-\u2013\u2014]?\s*(?:&#8212;\s*)?'
H_START = re.compile(sp('Item') + r'\s*1' + SEP + sp('Business'), re.I)
H_END = re.compile(sp('Item') + r'\s*1\s*A' + SEP + sp('Risk'), re.I)

lock = threading.Lock()
stat = {'ok': 0, 'fail': 0, 'bytes': 0}


def strip_html(b):
    t = b.decode('utf-8', 'ignore')
    t = re.sub(r'(?is)<(script|style).*?</\1>', ' ', t)
    t = re.sub(r'(?s)<[^>]+>', ' ', t)
    t = (t.replace('&#8217;', "'").replace('&#8220;', '"').replace('&#8221;', '"')
         .replace('&#8212;', '-').replace('&amp;', '&'))
    t = re.sub(r'&nbsp;?|&#160;|&#xa0;', ' ', t)
    return re.sub(r'\s+', ' ', t)


def primary_doc(cik, acc, rank=0):
    u = ('https://www.sec.gov/Archives/edgar/data/%d/%s/index.json'
         % (int(cik), acc.replace('-', '')))
    r = requests.get(u, headers=UA, timeout=60)
    if r.status_code != 200:
        return None
    items = json.loads(r.text)['directory']['item']
    cand = [it for it in items
            if it['name'].lower().endswith(('.htm', '.html', '.txt'))
            and not EXH.search(it['name']) and 'index' not in it['name'].lower()]
    if not cand:
        return None
    cand.sort(key=lambda it: int(it.get('size') or 0), reverse=True)
    if rank >= len(cand):
        return None
    return ('https://www.sec.gov/Archives/edgar/data/%d/%s/%s'
            % (int(cik), acc.replace('-', ''), cand[rank]['name']))


def best_section(txt):
    """The longest Item 1 ... Item 1A section in the text, and its length."""
    best, blen = None, -1
    for m in H_START.finditer(txt):
        st = m.end()
        en = H_END.search(txt[st:])
        body = txt[st:st + (en.start() if en else 14000)]
        if len(body) > blen:
            best, blen = body, len(body)
    return best, blen


def item1(cik, acc, rank=0):
    """Stream the 10-K and stop once a substantial Item 1 has been seen.

    `rank` picks the rank-th largest candidate document. A filing occasionally has a
    bigger attachment than the annual report itself, so a failure on the largest is
    retried on the next one down rather than given up on.
    """
    url = primary_doc(cik, acc, rank)
    if not url:
        return None, 'no primary document'
    rr = requests.get(url, headers=UA, timeout=240, stream=True)
    if rr.status_code != 200:
        return None, 'doc HTTP %d' % rr.status_code
    buf = b''
    got_end = False
    for chunk in rr.iter_content(262144):
        buf += chunk
        if len(buf) > MAXBYTES:
            break
        if len(buf) % 1048576 < 262144:      # check about once a megabyte
            # A table of contents contains both "Item 1. Business" and "Item 1A. Risk
            # Factors", so the stream stops only once a substantial section has been seen.
            if best_section(strip_html(buf))[1] >= 2500:
                got_end = True
                break
    rr.close()
    with lock:
        stat['bytes'] += len(buf)
    txt = strip_html(buf)
    if not H_START.search(txt):
        return None, 'no Item 1 header in %d KB' % (len(buf) // 1024)
    # Several matches: the table of contents, the running header, the body. Take the
    # one that yields the LONGEST section - the earlier ones are a line or two.
    best, blen = best_section(txt)
    if blen < 400:
        return None, 'Item 1 section too short (%d chars)' % blen
    return best.strip()[:14000], 'ok %d chars%s' % (blen, '' if got_end else ' (capped)')


def work(rec):
    body, note = None, 'not attempted'
    for rank in (0, 1, 2):
        try:
            body, note = item1(rec['cik'], rec['acc'], rank)
        except Exception as e:
            body, note = None, repr(e)[:120]
        if body:
            if rank:
                note += ' (doc #%d)' % (rank + 1)
            break
    with lock:
        stat['ok' if body else 'fail'] += 1
    return {'gvkey': rec['gvkey'], 'name': rec['name'], 'cik': int(rec['cik']),
            'acc': rec['acc'], 'note': note, 'business_text': body or ''}

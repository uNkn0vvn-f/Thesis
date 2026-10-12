"""Fetch each 485BPOS filing's SGML header and extract the authoritative
series (fund) / class (share class) roster.

A single 485BPOS covers a whole trust, so it can update many funds at once.
EDGAR's own header solves the fund-level mapping problem: every N-1A filing
carries SERIES-AND-CLASSES-CONTRACTS-DATA with, per fund,

    SERIES-ID              S000005466   permanent SEC fund id, survives renames
    SERIES-NAME            DWS Emerging Markets Fixed Income Fund
    CLASS-CONTRACT-ID      C000014875   per share class
    CLASS-CONTRACT-NAME    Class A
    CLASS-CONTRACT-TICKER  SZEAX        links to CRSP / Morningstar

The header file is ~14 KB, so this is cheap next to the prospectuses themselves.
Output: one gzipped shard per month, one record per filing.
"""
import csv, glob, gzip, json, os, re, threading, time
from concurrent.futures import ThreadPoolExecutor
import requests

from src import paths as tp  # noqa: E402

SRC = os.environ.get('SRC')
LISTS = ([SRC] if SRC else
         [tp.data('edgar', name) for name in ('485bpos_2020_2023.csv', '485bpos_2024_2026.csv')])
DEST = os.environ.get('DEST', tp.data('485bpos_headers'))
RATE = float(os.environ.get('RATE', '5.0'))
WORKERS = int(os.environ.get('WORKERS', '8'))
LIMIT = int(os.environ.get('LIMIT', '0'))
H = {'User-Agent': tp.sec_user_agent(),
     'Accept-Encoding': 'gzip, deflate', 'Host': 'www.sec.gov'}

_lk = threading.Lock(); _next = [0.0]
SERIES_BLOCK = re.compile(r'<SERIES>(.*?)</SERIES>', re.S | re.I)
CLASS_BLOCK = re.compile(r'<CLASS-CONTRACT>(.*?)</CLASS-CONTRACT>', re.S | re.I)


def tag(block, name):
    """The value of one SGML tag inside a header block, or an empty string."""
    m = re.search(r'<%s>([^\r\n<]*)' % name, block, re.I)
    return m.group(1).strip() if m else ''


def throttle():
    """Wait so that requests stay within RATE per second across all threads."""
    with _lk:
        now = time.time()
        wait = max(0.0, _next[0] - now)
        _next[0] = max(now, _next[0]) + 1.0 / RATE
    if wait:
        time.sleep(wait)


def parse_series(text):
    """The series (funds) and share classes listed in a filing's SGML header."""
    out = []
    for sb in SERIES_BLOCK.findall(text):
        classes = [{'class_id': tag(cb, 'CLASS-CONTRACT-ID'),
                    'class_name': tag(cb, 'CLASS-CONTRACT-NAME'),
                    'ticker': tag(cb, 'CLASS-CONTRACT-TICKER-SYMBOL')}
                   for cb in CLASS_BLOCK.findall(sb)]
        out.append({'series_id': tag(sb, 'SERIES-ID'),
                    'series_name': tag(sb, 'SERIES-NAME'),
                    'owner_cik': tag(sb, 'OWNER-CIK'),
                    'classes': classes})
    return out


def fetch(r, tries=3):
    """Download one filing's header page and return its fund roster, retrying failed
    requests."""
    acc = r['path'].split('/')[-1].replace('.txt', '')
    url = ('https://www.sec.gov/Archives/edgar/data/%s/%s/%s-index-headers.html'
           % (r['cik'], acc.replace('-', ''), acc))
    m = {'accession': acc, 'cik': r['cik'], 'company': r['company'], 'date': r['date']}
    for a in range(tries):
        try:
            throttle()
            resp = requests.get(url, headers=H, timeout=120)
            if resp.status_code != 200:
                if a < tries - 1:
                    time.sleep(2 * (a + 1)); continue
                return {**m, 'err': 'HTTP %d' % resp.status_code}
            series = parse_series(resp.text)
            return {**m, 'n_series': len(series),
                    'n_classes': sum(len(s['classes']) for s in series),
                    'series': series, 'err': None}
        except Exception as e:
            if a < tries - 1:
                time.sleep(2 * (a + 1)); continue
            return {**m, 'err': repr(e)[:200]}


def load_done():
    """Accessions already fetched without error, read from the existing shards."""
    done = set()
    for f in sorted(glob.glob(os.path.join(DEST, '*.jsonl.gz'))):
        try:
            with gzip.open(f, 'rt', encoding='utf-8') as fh:
                while True:
                    try:
                        l = fh.readline()
                    except (EOFError, OSError):
                        break
                    if not l:
                        break
                    try:
                        d = json.loads(l)
                        if not d.get('err'):
                            done.add(d['accession'])
                    except Exception:
                        pass
        except Exception:
            pass
    return done


if __name__ == '__main__':
    os.makedirs(DEST, exist_ok=True)
    rows = [r for path in LISTS for r in csv.DictReader(open(path, encoding='utf-8'))]
    if LIMIT:
        rows = rows[:LIMIT]
    # Fetch only the filings not yet in the shards, written to one shard per filing month.
    done = load_done()
    todo = [r for r in rows if r['path'].split('/')[-1].replace('.txt', '') not in done]
    print('total=%d already=%d to_fetch=%d' % (len(rows), len(done), len(todo)), flush=True)
    handles = {}
    t0 = time.time(); ok = err = nser = 0
    try:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for i, rec in enumerate(ex.map(fetch, todo), 1):
                p = os.path.join(DEST, 'hdr_%s_%s.jsonl.gz' % (rec['date'][:4], rec['date'][5:7]))
                if p not in handles:
                    handles[p] = gzip.open(p, 'at', encoding='utf-8')
                handles[p].write(json.dumps(rec, ensure_ascii=False) + '\n')
                if rec.get('err'):
                    err += 1
                else:
                    ok += 1; nser += rec['n_series']
                if i % 500 == 0:
                    for h in handles.values():
                        h.flush()
                    el = time.time() - t0
                    print('  %d/%d %.0fmin ok=%d err=%d series=%d ETA %.1f min'
                          % (i, len(todo), el / 60, ok, err, nser, (len(todo) - i) * (el / i) / 60), flush=True)
    finally:
        for h in handles.values():
            h.close()
    print('DONE %.1f min ok=%d err=%d  %d series rows total' % ((time.time() - t0) / 60, ok, err, nser), flush=True)

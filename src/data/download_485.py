"""Download the 485BPOS prospectuses of the filing lists into THESIS_DATA/485bpos.

The lists are those of src.data.filing_index (both, unless SRC names one).

Design notes
------------
* Stores the FULL extracted prospectus text, gzipped, sharded one file per month, so
  that a change in the extraction logic (Part C, strategy boundaries, keyword list)
  never needs a new download: everything derived is recomputable from these shards.
* Part C is NOT cut here - the offset is recorded instead, so the decision stays
  reversible without touching the network.
* lxml + itertext() rather than BeautifulSoup: 4.3x faster, verified to produce
  identical AI keyword counts on sample documents.
* Resumable: a later run skips accessions already present in the shards.
* Rate limited to 6 req/s (SEC fair-access allows 10).
"""
import csv, gzip, json, os, re, sys, threading, time, warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import requests
import lxml.html

from src import paths as tp  # noqa: E402

SRC = os.environ.get('SRC')
LISTS = ([SRC] if SRC else
         [tp.data('edgar', name) for name in ('485bpos_2020_2023.csv', '485bpos_2024_2026.csv')])
DEST = os.environ.get('DEST', tp.data('485bpos'))
RATE = float(os.environ.get('RATE', '6.0'))
WORKERS = int(os.environ.get('WORKERS', '12'))
LIMIT = int(os.environ.get('LIMIT', '0'))          # 0 = everything
H = {'User-Agent': tp.sec_user_agent(),
     'Accept-Encoding': 'gzip, deflate', 'Host': 'www.sec.gov'}

PARTC = re.compile(r'^[^\n]{0,40}?(part\s+c\b[^\n]{0,50}|item\s+2[5-9]\.?\s+exhibits?\b[^\n]{0,40})$', re.I | re.M)
AI_PHRASE = re.compile(
    r'artificial\s+intelligence|machine[\s-]?learning|deep\s+learning|neural\s+net|'
    r'natural\s+language\s+processing|large\s+language\s+model|generative\s+ai|'
    r'genetic\s+algorithm|evolutionary\s+comput|reinforcement\s+learning|chatgpt', re.I)
AI_ACRO = re.compile(r'\bAI\b|\bA\.I\.|\bML\b|\bLLMs?\b|\bNLP\b')

_lk = threading.Lock(); _next = [0.0]
_wlk = defaultdict(threading.Lock)
stats = {'ok': 0, 'err': 0, 'bytes': 0, 'chars': 0}
_slk = threading.Lock()


def throttle():
    with _lk:
        now = time.time()
        wait = max(0.0, _next[0] - now)
        _next[0] = max(now, _next[0]) + 1.0 / RATE
    if wait:
        time.sleep(wait)


def get(url):
    throttle()
    return requests.get(url, headers=H, timeout=300)


def to_text(content):
    d = lxml.html.fromstring(content)
    for x in d.xpath('//script|//style'):
        x.getparent().remove(x)
    return re.sub(r'\n{3,}', '\n\n', '\n'.join(d.itertext()))


def partc_offset(t):
    floor = int(0.40 * len(t))
    for m in PARTC.finditer(t):
        if m.start() >= floor:
            return m.start()
    return -1


def fetch(r, tries=3):
    acc = r['path'].split('/')[-1].replace('.txt', '')
    base = 'https://www.sec.gov/Archives/edgar/data/%s/%s' % (r['cik'], acc.replace('-', ''))
    m = {'accession': acc, 'cik': r['cik'], 'company': r['company'], 'date': r['date']}
    for a in range(tries):
        try:
            ix = get(base + '/index.json')
            if ix.status_code != 200:
                if a < tries - 1:
                    time.sleep(2 * (a + 1)); continue
                return {**m, 'err': 'index HTTP %d' % ix.status_code}
            items = ix.json()['directory']['item']
            cands = [it for it in items
                     if it['name'].lower().endswith(('.htm', '.html'))
                     and not re.search(r'^(0001|R\d|.*-index)', it['name'])
                     and not re.match(r'.*ex[-_ ]?\d', it['name'], re.I)]
            if not cands:
                # Old-style submissions carry no separate primary .htm - only the index.
                # Fall back to the complete .txt submission and pull the 485BPOS document
                # out of its SGML wrapper.
                full = get('https://www.sec.gov/Archives/' + r['path'])
                if full.status_code != 200:
                    return {**m, 'err': 'no candidate htm; txt HTTP %d' % full.status_code}
                doc = None
                for chunk in re.findall(r'<DOCUMENT>(.*?)</DOCUMENT>', full.text, re.S):
                    if '<TYPE>485BPOS' in chunk[:400]:
                        j = chunk.find('<TEXT>')
                        doc = chunk[j + 6:] if j >= 0 else chunk
                        break
                if doc is None:
                    return {**m, 'err': 'no 485BPOS document in submission'}
                t = to_text(doc.encode('utf-8', 'replace'))
                with _slk:
                    stats['bytes'] += len(full.content); stats['chars'] += len(t)
                return {**m, 'doc': '(sgml fallback)', 'html_bytes': len(full.content), 'doc_len': len(t),
                        'partc_offset': partc_offset(t),
                        'n_ai_phrase': len(AI_PHRASE.findall(t)), 'n_ai_acro': len(AI_ACRO.findall(t)),
                        'text': t, 'err': None}
            # index.json sometimes carries size as an empty string; int('') raises and
            # would lose the whole filing to the retry path.
            def _size(it):
                try:
                    return int(it.get('size') or 0)
                except (TypeError, ValueError):
                    return 0
            best = max(cands, key=_size)
            d = get(base + '/' + best['name'])
            if d.status_code != 200:
                if a < tries - 1:
                    time.sleep(2 * (a + 1)); continue
                return {**m, 'doc': best['name'], 'err': 'doc HTTP %d' % d.status_code}
            t = to_text(d.content)
            with _slk:
                stats['bytes'] += len(d.content); stats['chars'] += len(t)
            return {**m, 'doc': best['name'], 'html_bytes': len(d.content), 'doc_len': len(t),
                    'partc_offset': partc_offset(t),
                    'n_ai_phrase': len(AI_PHRASE.findall(t)), 'n_ai_acro': len(AI_ACRO.findall(t)),
                    'text': t, 'err': None}
        except Exception as e:
            if a < tries - 1:
                time.sleep(2 * (a + 1)); continue
            return {**m, 'err': repr(e)[:200]}


def shard_path(date):
    return os.path.join(DEST, '485bpos_%s_%s.jsonl.gz' % (date[:4], date[5:7]))


def load_done():
    done = set()
    if not os.path.isdir(DEST):
        return done
    for f in sorted(os.listdir(DEST)):
        if not f.endswith('.jsonl.gz'):
            continue
        n = 0
        try:
            with gzip.open(os.path.join(DEST, f), 'rt', encoding='utf-8') as fh:
                # Iterate defensively: an interrupted run leaves a truncated gzip tail.
                # Keep every record read before the break instead of discarding the shard,
                # otherwise a single bad tail silently re-downloads a whole month and
                # appends duplicate lines.
                while True:
                    try:
                        l = fh.readline()
                    except (EOFError, OSError) as e:
                        print('  note: %s has a truncated tail (%s); kept %d records' % (f, type(e).__name__, n))
                        break
                    if not l:
                        break
                    try:
                        d = json.loads(l)
                        if not d.get('err'):
                            done.add(d['accession']); n += 1
                    except Exception:
                        pass
        except Exception as e:
            print('  warning: could not open %s (%r)' % (f, e))
    return done


if __name__ == '__main__':
    os.makedirs(DEST, exist_ok=True)
    rows = [r for path in LISTS for r in csv.DictReader(open(path, encoding='utf-8'))]
    if LIMIT:
        rows = rows[:LIMIT]
    print('scanning existing shards in %s ...' % DEST, flush=True)
    done = load_done()
    todo = [r for r in rows if r['path'].split('/')[-1].replace('.txt', '') not in done]
    print('total=%d  already_done=%d  to_fetch=%d  workers=%d rate=%.1f/s'
          % (len(rows), len(done), len(todo), WORKERS, RATE), flush=True)
    if not todo:
        sys.exit(0)

    handles = {}
    t0 = time.time()
    try:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for i, rec in enumerate(ex.map(fetch, todo), 1):
                p = shard_path(rec['date'])
                if p not in handles:
                    handles[p] = gzip.open(p, 'at', encoding='utf-8')
                handles[p].write(json.dumps(rec, ensure_ascii=False) + '\n')
                if rec.get('err'):
                    stats['err'] += 1
                else:
                    stats['ok'] += 1
                if i % 100 == 0:
                    for h in handles.values():
                        h.flush()
                    el = time.time() - t0
                    print('  %d/%d  %.0fmin  ok=%d err=%d  %.1f GB down  %.2f s/filing  ETA %.1f h'
                          % (i, len(todo), el / 60, stats['ok'], stats['err'], stats['bytes'] / 1e9,
                             el / i, (len(todo) - i) * (el / i) / 3600), flush=True)
    finally:
        for h in handles.values():
            h.close()
    el = time.time() - t0
    print('DONE %.1f h  ok=%d err=%d  downloaded %.1f GB  extracted %.1f GB of text'
          % (el / 3600, stats['ok'], stats['err'], stats['bytes'] / 1e9, stats['chars'] / 1e9), flush=True)
    print('shards in %s' % DEST, flush=True)

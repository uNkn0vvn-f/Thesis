"""Split each 485BPOS prospectus into FUND-level records.

One 485BPOS covers a whole trust, so several funds' strategy text sits in one document.
The split has two layers:

  layer 1  EDGAR's own header gives the authoritative roster of funds in the filing
           (SERIES-ID / SERIES-NAME / share classes / tickers) - see fetch_headers.py
  layer 2  those SERIES-NAMEs are used as anchors in the prospectus text; each
           "Principal Investment Strategies" block is attributed to the nearest
           preceding anchor.

Layer 2 is heuristic, so the script reports an attribution rate rather than assuming
success. Records where attribution fails keep `series_id = None` and the filing-level
text, so nothing is silently dropped.

Runs entirely on local shards - no network.
"""
import glob, gzip, json, os, re, sys, collections

from src import paths as tp  # noqa: E402

TEXT_DIR = os.environ.get('TEXT_DIR', tp.data('485bpos'))
HDR_DIR = os.environ.get('HDR_DIR', tp.data('485bpos_headers'))
OUT_DIR = os.environ.get('OUT_DIR', tp.data('485bpos_funds'))
CUT_PARTC = os.environ.get('CUT_PARTC', '1') == '1'

# Heading wording varies by filer. AllianceBernstein writes "PRINCIPAL STRATEGIES",
# most others "Principal Investment Strategies"; some append "and Risks". Requiring the
# word "Investment" silently dropped every AB prospectus, so keep it optional.
STRAT_HEAD = re.compile(
    r'^[^\n]{0,60}?principal\s+(investment\s+)?strateg(y|ies)[^\n]{0,60}$', re.I | re.M)
STRAT_END = re.compile(
    r'^[^\n]{0,60}?(principal\s+(investment\s+)?risks?\b|'
    r'principal\s+risk\s+factors?\b|risk/return\s+bar\s+chart)[^\n]{0,60}$', re.I | re.M)
AI_PHRASE = re.compile(
    r'artificial\s+intelligence|machine[\s-]?learning|deep\s+learning|neural\s+net|'
    r'natural\s+language\s+processing|large\s+language\s+model|generative\s+ai|'
    r'genetic\s+algorithm|evolutionary\s+comput|reinforcement\s+learning|chatgpt', re.I)
AI_ACRO = re.compile(r'\bAI\b|\bA\.I\.|\bML\b|\bLLMs?\b|\bNLP\b')
QUANT = re.compile(r'quantitativ|systematic\s+(strateg|approach|model|process|invest)|algorithmic|'
                   r'mathematical\s+model|statistical\s+(model|technique|analys)|factor\s+model|'
                   r'computer\s+model|rules?[\s-]based|proprietary\s+(model|algorithm)', re.I)


def read_shards(d):
    for p in sorted(glob.glob(os.path.join(d, '*.jsonl.gz'))):
        try:
            with gzip.open(p, 'rt', encoding='utf-8') as fh:
                while True:
                    try:
                        line = fh.readline()
                    except (EOFError, OSError):
                        break
                    if not line:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except Exception:
                        break
        except Exception:
            continue


def anchor_re(name):
    """Series name -> tolerant regex. Prospectuses re-wrap names across lines and
    vary punctuation, so collapse whitespace and make ® / ™ / commas optional."""
    parts = [re.escape(w) for w in re.split(r'\s+', name.strip()) if w]
    if not parts:
        return None
    body = r'[\s\u00ae\u2122,\.]*'.join(parts)
    return re.compile(body, re.I)


WORD = re.compile(r'[A-Za-z0-9]+')
MAX_ANCHOR_DIST = int(os.environ.get('MAX_ANCHOR_DIST', '40000'))


def find_anchors(text, series):
    """Sorted (offset, series index) for every occurrence of a series name.

    This replaces anchor_re above, which lets "Growth Fund" match
    inside "XYZ Large Growth Fund"; since attribution takes the NEAREST preceding
    anchor, the shorter and more generic name wins systematically.

    Word boundaries alone do NOT fix that, which is worth stating because it is the
    obvious patch and it fails: "Growth Fund" genuinely occurs as two whole words
    inside "XYZ Large Growth Fund", so a bounded regex matches it too. What fixes
    it is preferring the LONGEST name at any given place in the text - a match
    wholly contained in another fund's match is discarded.

    The document is tokenised once and each name looked up as a sequence of whole
    words, which also keeps the cost to one pass rather than one pass per fund: a
    trust prospectus can carry two hundred series against a two-megabyte document.
    """
    words, offs = [], []
    for m in WORD.finditer(text):
        words.append(m.group(0).lower())
        offs.append(m.end())
    index = {}
    for i, w in enumerate(words):
        index.setdefault(w, []).append(i)
    starts = [o - len(w) for o, w in zip(offs, words)]
    spans = []
    for si, s in enumerate(series):
        nm = WORD.findall((s.get('series_name') or '').lower())
        if not nm:
            continue
        k = len(nm)
        for i in index.get(nm[0], ()):
            if words[i:i + k] == nm:
                spans.append((starts[i], offs[i + k - 1], si))
    # longest-match-wins: walk left to right, longer span first at a tie, and drop
    # anything that ends inside a span already kept
    out, max_end = [], -1
    for st, en, si in sorted(spans, key=lambda t: (t[0], -t[1])):
        if en <= max_end:
            continue
        out.append((st, si))
        max_end = en
    return out


# Some prospectuses set headings in small caps, one element per letter. lxml's
# itertext() then yields "P / RINCIPAL / I / NVESTMENT / S / TRATEGIES", which no
# amount of heading-wording tolerance can match - it is a whitespace artifact of
# text extraction, not a wording variant. For those documents, search a copy with
# ALL whitespace removed and map the hit back to the original offset.
SQUASH_HEAD = re.compile(r'principal(investment)?strateg(y|ies)', re.I)
SQUASH_END = re.compile(r'principal(investment)?risks?', re.I)


def squash(t):
    """Whitespace-free copy plus index map back into the original string."""
    chars, idx = [], []
    for i, ch in enumerate(t):
        if not ch.isspace():
            chars.append(ch); idx.append(i)
    return ''.join(chars), idx


def strategy_spans(t):
    out = []
    for m in STRAT_HEAD.finditer(t):
        e = STRAT_END.search(t, m.end())
        stop = e.start() if e else min(m.end() + 12000, len(t))
        if 200 <= stop - m.start() <= 40000:
            out.append((m.start(), stop))
    if out:
        return out
    s, idx = squash(t)
    if not s:
        return out
    for m in SQUASH_HEAD.finditer(s):
        a = idx[m.start()]
        if out and a < out[-1][1]:
            continue                           # already inside the previous span
        e = SQUASH_END.search(s, m.end())
        stop = idx[e.start()] if e and e.start() < len(idx) else min(a + 12000, len(t))
        if 200 <= stop - a <= 40000:
            out.append((a, stop))
    return out


def segment(text, series):
    """Return list of (series_dict_or_None, strategy_text). Attribution = nearest
    preceding series-name occurrence before the strategy heading."""
    spans = strategy_spans(text)
    anchors = find_anchors(text, series)          # (position, series index)
    out, used = [], set()
    j, cur = 0, None
    for a, b in spans:
        # spans arrive in document order, so the anchor pointer only moves forward
        while j < len(anchors) and anchors[j][0] < a:
            cur = anchors[j]
            j += 1
        # with no distance cap, a trust-wide paragraph in the statutory
        # section is claimed by whatever fund was named last, however far back that
        # was. A shared "we may use artificial intelligence" then becomes a false
        # positive for exactly one fund. Beyond the cap the section is unattributed.
        best = cur[1] if (cur is not None and a - cur[0] <= MAX_ANCHOR_DIST) else None
        if best is not None:
            used.add(best)
            out.append((series[best], text[a:b]))
        else:
            out.append((None, text[a:b]))
    return out, used


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    hdr = {r['accession']: r for r in read_shards(HDR_DIR) if not r.get('err')}
    print('headers loaded: %d filings' % len(hdr), flush=True)

    st = collections.Counter()
    handles = {}
    nfilings = 0
    # A 485BPOS can name several co-registrants (e.g. one Putnam filing covers five
    # separate fund-company CIKs). EDGAR's form index lists such a filing once per
    # filer, so the same document was downloaded more than once. Same document, same
    # funds - keep the first and skip the rest, or the panel double-counts.
    seen_acc = set()
    try:
        for r in read_shards(TEXT_DIR):
            if r.get('err'):
                continue
            if r['accession'] in seen_acc:
                st['duplicate_filings_skipped'] += 1
                continue
            seen_acc.add(r['accession'])
            nfilings += 1
            t = r['text']
            if CUT_PARTC and r.get('partc_offset', -1) > 0:
                t = t[:r['partc_offset']]
            h = hdr.get(r['accession'])
            series = h['series'] if h else []
            st['filings'] += 1
            st['filings_no_header'] += 0 if h else 1
            st['series_in_header'] += len(series)

            pairs, used = segment(t, series)
            st['strategy_blocks'] += len(pairs)
            st['blocks_attributed'] += sum(1 for s, _ in pairs if s)
            st['blocks_unattributed'] += sum(1 for s, _ in pairs if not s)
            st['series_matched'] += len(used)
            st['series_unmatched'] += max(len(series) - len(used), 0)
            if not pairs:
                st['filings_no_strategy_block'] += 1

            # One record per (filing, fund). The summary section and the Item 9 detail
            # section both carry a "Principal Strategies" heading, so a fund typically
            # owns several blocks; concatenate them so each fund is one decision.
            grouped = collections.OrderedDict()
            for s, strat in pairs:
                key = (s or {}).get('series_id') or '__unattributed__'
                if key not in grouped:
                    grouped[key] = {'series': s, 'blocks': []}
                grouped[key]['blocks'].append(strat)
            # funds listed in the header that no block matched: keep them, empty text,
            # so the fund panel stays complete rather than silently losing rows
            for s in series:
                if s.get('series_id') and s['series_id'] not in grouped:
                    grouped[s['series_id']] = {'series': s, 'blocks': []}

            p = os.path.join(OUT_DIR, 'funds_%s_%s.jsonl.gz' % (r['date'][:4], r['date'][5:7]))
            if p not in handles:
                handles[p] = gzip.open(p, 'at', encoding='utf-8')
            for key, g in grouped.items():
                s = g['series']
                strat = '\n\n[...]\n\n'.join(g['blocks'])
                rec = {'accession': r['accession'], 'cik': r['cik'], 'company': r['company'],
                       'date': r['date'],
                       'series_id': (s or {}).get('series_id'), 'series_name': (s or {}).get('series_name'),
                       'tickers': [c['ticker'] for c in (s or {}).get('classes', []) if c.get('ticker')],
                       'n_classes': len((s or {}).get('classes', [])),
                       'n_blocks': len(g['blocks']), 'strat_len': len(strat),
                       'ai_strategy': bool(AI_PHRASE.search(strat)) or bool(AI_ACRO.search(strat)),
                       'quant_strategy': bool(QUANT.search(strat)),
                       'strategy': strat}
                handles[p].write(json.dumps(rec, ensure_ascii=False) + '\n')
                st['fund_records'] += 1
                st['ai_candidates'] += rec['ai_strategy']
                if not g['blocks']:
                    st['funds_without_text'] += 1
            if nfilings % 1000 == 0:
                for x in handles.values():
                    x.flush()
                print('  %d filings -> %d fund records' % (nfilings, st['fund_records']), flush=True)
    finally:
        for x in handles.values():
            x.close()

    b = st['strategy_blocks'] or 1
    s_tot = st['series_in_header'] or 1
    print('\n--- segmentation report ---')
    print('filings processed          : %d  (no header: %d)' % (st['filings'], st['filings_no_header']))
    print('duplicate filings skipped  : %d  (co-registrant rows in the EDGAR index)' % st['duplicate_filings_skipped'])
    print('series listed in headers   : %d' % st['series_in_header'])
    print('fund-level records written : %d' % st['fund_records'])
    print('strategy blocks found      : %d' % st['strategy_blocks'])
    print('  attributed to a series   : %d (%.1f%%)' % (st['blocks_attributed'], 100 * st['blocks_attributed'] / b))
    print('  unattributed             : %d (%.1f%%)' % (st['blocks_unattributed'], 100 * st['blocks_unattributed'] / b))
    print('series matched in text     : %d (%.1f%% of header series)' % (st['series_matched'], 100 * st['series_matched'] / s_tot))
    print('series never found in text : %d' % st['series_unmatched'])
    print('filings with no strategy   : %d' % st['filings_no_strategy_block'])
    print('funds with no strategy text: %d' % st['funds_without_text'])
    print('AI-keyword fund records    : %d (%.1f%%)' % (st['ai_candidates'], 100 * st['ai_candidates'] / max(st['fund_records'], 1)))
    print('\nshards in %s' % OUT_DIR)

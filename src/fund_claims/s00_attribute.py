"""Attribution: assign the AI text of a multi-fund filing to its fund where the segmenter could not.

A trust often files one 485BPOS for several of its funds. Where src.data.segment_funds cannot
split the Principal Investment Strategies text, it keeps the unassigned text in a record with
no series id, and the funds of that filing come out with no text. An AI claim in such text
would be invisible to the fund panel, and the as-of panel would score the fund's holdings
against an older prospectus. ROBO Global Artificial Intelligence ETF is an example: it files
every August, and its strategy text sits in the unassigned part of each filing from 2021 to
2026.

WHAT THIS DOES, AND WHAT IT DELIBERATELY DOES NOT.
  * Only filings whose unassigned text mentions AI are touched. Unassigned text with no AI
    in it is left exactly as it was: attributing it could only ADD explicit "no claim" rows,
    and a wrong attribution there manufactures a fund that stopped claiming AI.
  * The text is split at each "Principal Investment Strateg..." heading, and each block is
    scored against the candidate funds - those the SEC header registers for the filing and
    the segmenter gave no text - by the part of the fund's NAME that tells it from its
    siblings (ROBO Global "Artificial Intelligence" vs "Robotics and Automation"), and by
    any ticker that is not a word in a sibling's name.
  * STRICT: a name counts only if its distinguishing part has two words that are not
    ordinary fund vocabulary ("income", "value", "growth", "bond" ...); a block goes to a
    fund only on at least 3 hits and at least three times the runner-up. A single
    candidate is held to the same test - it must be NAMED in the block - rather than
    inheriting whatever text is left over, which is often the trust's shared section.
  * Blocks that fail stay in a remainder record with no series id, as before.

WHY NOT A BLIND ACCURACY TEST. Scored against the filings the segmenter did split, the
attribution reaches 47.5% precision, and the examples show why that test cannot be
trusted in either direction: a generic one-word name ("AB Income Fund" <- "income") did
cause real errors, which the strict rule above removes, but the segmenter's own attribution
often gives a fund its sibling's section ("Principal Investment Strategies of the Partners
Fund" inside the Cornerstone Fund's text), so a correct content match was scored as wrong.
The set this step changes is small enough to be read instead: the log lists every block it
attributes, with its hits against the chosen fund and against the best alternative, and the
first AI sentence in it.

OUTPUT. A mirror of the corpus directory, same shard names and format; the raw corpus is not
touched. Filled records carry `attributed='block'` and `ai_strategy` recomputed with s01's
AI_RE (it agrees with the upstream flag on 22,691 of 22,694 records checked).
"""
import collections, glob, gzip, io, json, os, re, sys

from src import paths as tp  # noqa: E402

OUT = tp.out('fund_claims')
LOGS = tp.logs('fund_claims')
os.makedirs(LOGS, exist_ok=True)

IN_DIR = tp.data('485bpos_funds')
HDR_DIR = tp.data('485bpos_headers')
DEST = os.path.join(OUT, 'funds_attributed')
os.makedirs(DEST, exist_ok=True)

from .s01_extract_evidence import AI_RE           # noqa: E402  one definition of "mentions AI"

HEAD_RE = re.compile(r'principal\s+investment\s+strateg', re.I)
MARK_RE = re.compile(r'\(R\)|\(TM\)|\(SM\)|[\u00ae\u2122\u2120]', re.I)
STOP = {'and', 'of', 'the', '&', 'for', 'a', 'an', 'in', 'on', 'to'}
TRAIL = {'etf', 'fund', 'funds', 'index', 'portfolio', 'shares', 'trust', 'class',
         'series', 'inc', 'llc', 'the'}
# ordinary fund vocabulary: a distinguishing part made only of these names nobody
GENERIC = TRAIL | {
    'income', 'value', 'growth', 'core', 'equity', 'equities', 'bond', 'bonds', 'total',
    'return', 'returns', 'balanced', 'strategic', 'opportunities', 'opportunity', 'select',
    'global', 'international', 'dividend', 'municipal', 'short', 'long', 'duration', 'high',
    'yield', 'term', 'intermediate', 'large', 'mid', 'small', 'cap', 'blend', 'market',
    'markets', 'emerging', 'us', 'u', 's', 'world', 'fixed', 'government', 'money',
    'allocation', 'conservative', 'moderate', 'aggressive', 'plus', 'premium', 'daily',
    'strategy', 'strategies', 'multi', 'asset', 'assets', 'focus', 'focused', 'active',
    'enhanced', 'quality', 'capital', 'appreciation', 'real', 'estate', 'credit', 'tax',
    'free', 'exempt', 'free', 'managed', 'target', 'retirement', 'institutional', 'investor',
    'admiral', 'advantage', 'partners', 'blue', 'fundamental', 'leaders', 'leadership',
    'sustainable', 'esg', 'impact', 'america', 'american', 'north', 'europe', 'asia',
    'pacific', 'dynamic', 'flexible', 'absolute', 'alternative', 'alternatives', 'rate',
    'floating', 'inflation', 'protected', 'securities', 'treasury', 'ultra', 'x', '2x', '3x',
    'bull', 'bear', 'long', 'short', 'option', 'options', 'covered', 'call'}
MIN_HITS, DOMINANCE = 3, 3.0
# a name found in more than max(4, a quarter of the blocks) is spread too thin to name one
# fund; a multi-fund filing often repeats each fund's sections once per share-class
# prospectus, so a real name can legitimately appear in two or three blocks per copy
SPREAD_MIN, SPREAD_SHARE = 4, 0.25


def tokens(name):
    s = MARK_RE.sub(' ', name or '')
    return [t for t in re.split(r'[^A-Za-z0-9&]+', s.lower()) if t]


def distinctive(names):
    """Per candidate, the tokens that tell it apart: strip the prefix and suffix every
    candidate shares, trailing generic words, and stop words."""
    toks = {k: tokens(v) for k, v in names.items()}
    ks = list(toks)
    if len(ks) > 1:
        pre = 0
        while all(len(toks[k]) > pre for k in ks) and len({toks[k][pre] for k in ks}) == 1:
            pre += 1
        suf = 0
        while (all(len(toks[k]) > pre + suf for k in ks)
               and len({toks[k][-1 - suf] for k in ks}) == 1):
            suf += 1
        toks = {k: t[pre:len(t) - suf] for k, t in toks.items()}
    out = {}
    for k, t in toks.items():
        while t and t[-1] in TRAIL:
            t = t[:-1]
        out[k] = [x for x in t if x not in STOP]
    return out


def phrase_re(tks):
    """The first two distinguishing tokens, allowing 'and' / 'of' / '&' between them - and
    only if at least one of the two is not ordinary fund vocabulary. A lone token is enough
    only when it is itself specific: not fund vocabulary and six letters or more
    ("technology", "kensho"; never "income" or "blue")."""
    t = tks[:2]
    if len(t) == 1:
        if t[0] in GENERIC or len(t[0]) < 6:
            return None
        return re.compile(r'\b' + re.escape(t[0]) + r'\b', re.I)
    if len(t) < 2 or all(x in GENERIC for x in t):
        return None
    sep = r'[\W_]+(?:(?:and|of|the|&)[\W_]+)*'
    return re.compile(r'\b' + sep.join(re.escape(x) for x in t) + r'\b', re.I)


def scorers(names, tickers):
    dist = distinctive(names)
    sibling_words = set(tokens(' '.join(names.values())))
    sc = {}
    for k in names:
        pats = []
        pr = phrase_re(dist.get(k) or [])
        if pr is not None:
            pats.append(pr)
        for tk in tickers.get(k) or []:
            tk = (tk or '').strip()
            if len(tk) >= 3 and tk.lower() not in sibling_words:
                pats.append(re.compile(r'\b' + re.escape(tk) + r'\b'))
        sc[k] = pats
    return sc


def split_blocks(text):
    idx = [m.start() for m in HEAD_RE.finditer(text)]
    if not idx:
        return [text]
    idx[0] = 0                              # any preamble travels with the first block
    idx.append(len(text))
    return [text[idx[i]:idx[i + 1]] for i in range(len(idx) - 1)]


def score(b, sc):
    return sorted(((sum(len(p.findall(b)) for p in pats), k) for k, pats in sc.items()),
                  reverse=True)


def headers():
    H = {}
    for p in sorted(glob.glob(os.path.join(HDR_DIR, '*.jsonl.gz'))):
        try:
            with gzip.open(p, 'rt', encoding='utf-8', errors='replace') as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    ser = [(s.get('series_id'), s.get('series_name'),
                            [c.get('ticker') for c in (s.get('classes') or [])
                             if c.get('ticker')])
                           for s in (r.get('series') or []) if s.get('series_id')]
                    if ser:
                        H[r.get('accession')] = ser
        except Exception:
            continue
    return H


def read_shard(p):
    recs = []
    try:
        with gzip.open(p, 'rt', encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    recs.append(json.loads(line))
                except Exception:
                    break                   # a truncated shard ends here, as in s01
    except Exception:
        pass
    return recs


def has_text(r):
    return bool((r.get('strategy') or '').strip())


def main():
    buf = io.StringIO()

    def w(s=''):
        buf.write(s + '\n')
        try:
            print(s)
        except UnicodeEncodeError:
            print(s.encode('ascii', 'replace').decode())

    H = headers()
    C = collections.Counter()
    shown = []

    for p in sorted(glob.glob(os.path.join(IN_DIR, '*.jsonl.gz'))):
        recs = read_shard(p)
        by_acc = collections.defaultdict(list)
        for i, r in enumerate(recs):
            by_acc[r.get('accession')].append(i)
        drop, extra = set(), []

        for acc, ix in by_acc.items():
            orphans = [i for i in ix if not (recs[i].get('series_id') or '').strip()
                       and has_text(recs[i])]
            if not orphans:
                continue
            C['filings with unassigned text'] += 1
            text = '\n'.join(recs[i]['strategy'] for i in orphans)
            if not AI_RE.search(text):
                C['  ... no AI in it: left untouched'] += 1
                continue
            C['  ... mentioning AI'] += 1
            if len(H.get(acc) or []) == 1:
                # one registered fund: attribution is certain and s01's header recovery
                # already makes it, on the unaltered text
                C['      single-fund filing, left to s01'] += 1
                continue
            base = recs[orphans[0]]
            have = {(recs[i].get('series_id') or '').strip() for i in ix
                    if (recs[i].get('series_id') or '').strip() and has_text(recs[i])}
            empty = {(recs[i].get('series_id') or '').strip(): i for i in ix
                     if (recs[i].get('series_id') or '').strip() and not has_text(recs[i])}
            if acc in H:
                cand = [(s, n, t) for s, n, t in H[acc] if s not in have]
            else:
                cand = [(s, recs[i].get('series_name'), recs[i].get('tickers') or [])
                        for s, i in empty.items()]
            if not cand:
                C['      no candidate fund'] += 1
                continue
            sc = scorers({s: n for s, n, _ in cand}, {s: t for s, _, t in cand})
            blocks = split_blocks(text)
            # A fund's own name sits in its own one or two sections. A word that turns up
            # all over the text - the sponsor ("State Street"), a market term ("investment
            # grade", "intrinsic value") - is not naming anybody, however specific it looks.
            spread_cap = max(SPREAD_MIN, int(SPREAD_SHARE * len(blocks)))
            for k in list(sc):
                spread = sum(1 for b in blocks if any(p.search(b) for p in sc[k]))
                if spread > spread_cap:
                    sc[k] = []
                    C['      candidate names dropped as not distinctive here'] += 1
            if not any(sc.values()):
                C['      no candidate has a usable name or ticker'] += 1
                continue

            got, un = collections.defaultdict(list), []
            for b in blocks:
                s = score(b, sc)
                best, second = s[0], (s[1][0] if len(s) > 1 else 0)
                if best[0] >= MIN_HITS and best[0] >= DOMINANCE * second:
                    got[best[1]].append(b)
                    if AI_RE.search(b):
                        m = AI_RE.search(b)
                        a = max(0, b.rfind('.', 0, m.start()) + 1)
                        z = b.find('.', m.end())
                        shown.append((str(base.get('date'))[:10],
                                      dict((x, n) for x, n, _ in cand)[best[1]],
                                      best[0], second, len(cand),
                                      re.sub(r'\s+', ' ', b[a:z if z > 0 else None])[:180]))
                else:
                    un.append(b)
            # If AI text is still unassigned, a fund given only AI-free blocks may simply
            # not have been given ITS AI block - and emitting that partial text would record
            # a fund that stopped claiming AI. Only a certain zero is written: those funds'
            # blocks go back to the remainder.
            if any(AI_RE.search(b) for b in un):
                for sid in [s for s, bl in got.items()
                            if not any(AI_RE.search(b) for b in bl)]:
                    un.extend(got.pop(sid))
                    C['      AI-free text withheld, AI still unassigned'] += 1
            if not got:
                # nothing named: the records stay byte-for-byte as they were, so their
                # evidence hash - and the labels already paid for - still match
                C['      nothing attributable, left untouched'] += 1
                continue

            meta = {s: (n, t) for s, n, t in cand}
            for sid, blocks in got.items():
                txt = '\n'.join(blocks)
                if sid in empty:
                    r = dict(recs[empty[sid]])
                    drop.add(empty[sid])
                else:
                    r = {k: base.get(k) for k in ('accession', 'cik', 'company', 'date')}
                    r['series_id'], r['series_name'] = sid, meta[sid][0]
                    r['tickers'] = meta[sid][1]
                r.update(strategy=txt, strat_len=len(txt), n_blocks=len(blocks),
                         ai_strategy=bool(AI_RE.search(txt)), quant_strategy=None,
                         attributed='block')
                extra.append(r)
                C['      funds given text'] += 1
                if r['ai_strategy']:
                    C['      ... whose text mentions AI'] += 1
            for i in orphans:
                drop.add(i)
            if un:
                rem = dict(base)
                txt = '\n'.join(un)
                rem.update(strategy=txt, strat_len=len(txt), n_blocks=len(un),
                           ai_strategy=bool(AI_RE.search(txt)), attributed='remainder')
                extra.append(rem)
                if rem['ai_strategy']:
                    C['      remainder still mentioning AI, no series'] += 1

        out = [r for i, r in enumerate(recs) if i not in drop] + extra
        with tp.gzip_text_writer(os.path.join(DEST, os.path.basename(p))) as fh:
            for r in out:
                fh.write(json.dumps(r) + '\n')

    w('=' * 96)
    w('ATTRIBUTING UNASSIGNED AI TEXT TO THE FUND A MULTI-FUND FILING NAMES')
    w('=' * 96)
    w('  headers read : %d filings' % len(H))
    for k, v in C.items():
        w('  %-52s: %d' % (k, v))
    w()
    w('  EVERY AI-BEARING BLOCK ATTRIBUTED - read these, they are the whole change')
    w('  %-10s %-52s %5s %5s %5s  first AI sentence' % ('filed', 'fund', 'hits', 'next',
                                                      'cands'))
    for d, nm, h, s2, nc, snt in sorted(shown):
        w('  %-10s %-52s %5d %5d %5d  %s' % (d, (nm or '')[:52], h, s2, nc, snt))
    w()
    w('wrote %s' % DEST)
    with io.open(os.path.join(LOGS, 's00_attribute.txt'), 'w', encoding='utf-8') as fh:
        fh.write(buf.getvalue())


if __name__ == '__main__':
    main()

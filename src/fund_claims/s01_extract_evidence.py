"""Evidence extraction: a deterministic, frozen evidence block per filing.

A character window around each AI keyword makes the classifier input depend on where
a paragraph sits in the prospectus: two windows of different width show two runs different
text about the same fund. The evidence block is therefore built from sentences, not
character offsets. This extracts, deterministically:

  * every sentence in the Principal Investment Strategies text containing an AI token,
  * one sentence on each side of it for context, merged into contiguous runs,
  * plus the fund's 80%-of-assets policy sentence, which is where the inclusion test
    ("derives 50% of revenue from...", "as defined by Indxx, LLC", "the portfolio
    managers believe...") is stated.

The block is a function of the sentences only. Editing an unrelated paragraph, or
inserting text that shifts every offset, leaves it byte-identical. Its sha256 is stored
so that reproducibility can be verified later, which a character window does not allow.

Sentence splitting is abbreviation-safe: prospectuses are full of "U.S.", "Inc.",
"S&P 500", "e.g.", "A.I." and "50.5%", and a naive split on [.!?] breaks on all of them.

SIBLINGS' TEXT IS REMOVED FIRST (sibling.py). Where the segmenter hands one fund a trust's
shared section, the other funds' AI sentences would become that fund's claim (TrueShares
Eagle Global Renewable Energy Income ETF would read as an AI developer from a sibling AI
ETF's 80% policy). A sentence about another series of the same filing, or inside a section
headed with another series' name, is dropped before the block is built. The log lists every
fund-filing that lost sentences.
"""
import collections, glob, gzip, hashlib, io, json, os, re, sys

from src import paths as tp  # noqa: E402

OUT = tp.out('fund_claims')
LOGS = tp.logs('fund_claims')
os.makedirs(OUT, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)

# s00_attribute gives each fund in a multi-fund filing its own strategy text; the raw
# corpus left 6,887 filings' text with no series id and those funds with none.
IN_DIR = os.environ.get('IN_DIR', os.path.join(OUT, 'funds_attributed'))
HDR_DIR = os.environ.get('HDR_DIR', tp.data('485bpos_headers'))
DEST = os.path.join(OUT, 'evidence.jsonl.gz')
CONTEXT = int(os.environ.get('CONTEXT', '1'))      # sentences of context each side
POLICY_NEAR = int(os.environ.get('POLICY_NEAR', '3'))   # policy sentence must be this
                                                        # close to an AI mention
MAX_CHARS = int(os.environ.get('MAX_CHARS', '12000'))
from .sibling import own_sentences                      # noqa: E402

AI_RE = re.compile(
    r'artificial\s+intelligence|machine[\s-]?learning|deep\s+learning|neural\s+net|'
    r'natural\s+language\s+processing|large\s+language\s+model|generative\s+ai|'
    r'genetic\s+algorithm|evolutionary\s+comput|reinforcement\s+learning|chatgpt|'
    r'\bAI\b|\bA\.I\.|\bML\b|\bLLMs?\b|\bNLP\b', re.I)

# the Names Rule policy sentence, where the inclusion test is stated
POLICY_RE = re.compile(r'\b80\s?%|\b80\s+percent\b', re.I)

# --------------------------------------------------------------------------- #
# abbreviation-safe sentence splitting
# --------------------------------------------------------------------------- #
_ABBR = (r'U\.S\.A?|U\.K\.|E\.U\.|A\.I\.|e\.g\.|i\.e\.|etc\.|vs\.|cf\.|'
         r'Inc\.|Corp\.|Co\.|Ltd\.|L\.P\.|L\.L\.C\.|LLC\.|Plc\.|N\.A\.|'
         r'Mr\.|Mrs\.|Ms\.|Dr\.|Jr\.|Sr\.|St\.|No\.|Nos\.|Fig\.|Sec\.|'
         r'Jan\.|Feb\.|Mar\.|Apr\.|Jun\.|Jul\.|Aug\.|Sep\.|Sept\.|Oct\.|Nov\.|Dec\.|'
         r'S&P|Reg\.|Art\.|Rev\.|approx\.|Ave\.|Rd\.')
ABBR_RE = re.compile(r'\b(?:%s)' % _ABBR)
SENT_END = re.compile(r'(?<=[.!?])[\s\u00a0]+')
DOT = '\x00'


def split_sentences(text):
    """Split into sentences without breaking on abbreviations, initials or decimals."""
    t = re.sub(r'[\r\n\u00a0\t]+', ' ', text)
    t = re.sub(r' {2,}', ' ', t).strip()
    if not t:
        return []
    # protect abbreviations
    t = ABBR_RE.sub(lambda m: m.group(0).replace('.', DOT), t)
    # protect decimals and numbered lists like "(1)." and single initials "J. P."
    t = re.sub(r'(?<=\d)\.(?=\d)', DOT, t)
    t = re.sub(r'\b([A-Z])\.(?=\s*[A-Z]\.)', r'\1' + DOT, t)
    parts = [p for p in SENT_END.split(t) if p.strip()]
    return [p.replace(DOT, '.').strip() for p in parts]


def build_block(strategy, own_name=None, siblings=(), dropped=None, extra_drop=None):
    """Return (block_text, n_ai_sentences, n_sentences_total) or (None, 0, n).
    Sentences belonging to a sibling series are removed first; if `dropped` is a list,
    the removed AI sentences are appended to it."""
    sents = split_sentences(strategy or '')
    if not sents:
        return None, 0, 0
    if siblings:
        mask = own_sentences(sents, own_name, siblings)
        if extra_drop is not None:
            mask = [k and not extra_drop(s) for s, k in zip(sents, mask)]
        if dropped is not None:
            dropped.extend(s for s, k in zip(sents, mask) if not k and AI_RE.search(s))
        sents = [s for s, k in zip(sents, mask) if k]
        if not sents:
            return None, 0, 0
    # A single 485BPOS document often registers many funds and repeats whole
    # paragraphs verbatim. Dropping exact repeats keeps the block about the claim
    # rather than about how many times the filer restated it.
    seen, uniq = set(), []
    for s in sents:
        k = re.sub(r'\W+', '', s.lower())
        if k and k in seen:
            continue
        seen.add(k)
        uniq.append(s)
    sents = uniq
    hit = [i for i, s in enumerate(sents) if AI_RE.search(s)]
    if not hit:
        return None, 0, len(sents)
    keep = set()
    for i in hit:
        for j in range(max(0, i - CONTEXT), min(len(sents), i + CONTEXT + 1)):
            keep.add(j)
    # The 80%-of-assets sentence carries the inclusion test - but only the one that
    # belongs to THIS claim. A multi-fund prospectus contains one per fund, and
    # taking them all floods the block with other funds' boilerplate: before this
    # restriction, 13% of blocks were more than half policy text and one carried 61
    # such sentences. Keep only those sitting near an AI mention.
    for i, s in enumerate(sents):
        if POLICY_RE.search(s) and any(abs(i - h) <= POLICY_NEAR for h in hit):
            keep.add(i)
    idx = sorted(keep)
    # merge into contiguous runs so the block reads as prose, with a marker at gaps
    chunks, run = [], [idx[0]]
    for k in idx[1:]:
        if k == run[-1] + 1:
            run.append(k)
        else:
            chunks.append(run)
            run = [k]
    chunks.append(run)
    block = '\n\n[...]\n\n'.join(' '.join(sents[k] for k in c) for c in chunks)
    return block[:MAX_CHARS], len(hit), len(sents)


def single_series_by_accession():
    """accession -> series_id, for filings that register exactly ONE series.

    The upstream segmenter leaves series_id empty on 123 of the 1,817 AI-bearing
    filings. They would be lost in every fund-level groupby, and they are not
    junk: two of them are ROBO Global Artificial Intelligence ETF and WisdomTree
    Artificial Intelligence and Innovation Fund. The filing header carries the
    registered series list, so where a filing registers exactly one series the
    attribution is unambiguous and is recovered here. Where it registers several, the
    AI text cannot be assigned without re-parsing, and the row is kept with
    series_resolved = False so that it is visible rather than absent.
    """
    out = {}
    for p in sorted(glob.glob(os.path.join(HDR_DIR, '*.jsonl.gz'))):
        try:
            with gzip.open(p, 'rt', encoding='utf-8', errors='replace') as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    ser = r.get('series') or []
                    if len(ser) == 1 and ser[0].get('series_id'):
                        out[r.get('accession')] = (ser[0]['series_id'],
                                                   ser[0].get('series_name'))
        except Exception:
            continue
    return out


def series_by_accession():
    """accession -> every series name the filing header registers."""
    out = collections.defaultdict(list)
    for p in sorted(glob.glob(os.path.join(HDR_DIR, '*.jsonl.gz'))):
        try:
            with gzip.open(p, 'rt', encoding='utf-8', errors='replace') as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    for s in r.get('series') or []:
                        if s.get('series_name'):
                            out[r.get('accession')].append(s['series_name'])
        except Exception:
            continue
    return out


ADJ = tp.data('hand_input', 'sibling_adjudications.csv')


def norm(s):
    return re.sub(r'\W+', '', s.lower())


def load_adjudications():
    """series_id -> list of (action, sibling_series_id, pattern), from the hand-checked
    list of text a sibling's section left in this fund's (sibling_adjudications.csv)."""
    import csv
    out = collections.defaultdict(list)
    with io.open(ADJ, encoding='utf-8', newline='') as fh:
        for r in csv.DictReader(fh):
            out[r['series_id']].append((r['action'], r['sibling_series_id'],
                                        r['pattern'].strip().lower()))
    return out


def shards(d):
    for p in sorted(glob.glob(os.path.join(d, '*.jsonl.gz'))):
        try:
            with gzip.open(p, 'rt', encoding='utf-8') as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except Exception:
                        break
        except Exception:
            continue


def main():
    buf = io.StringIO()

    def w(s=''):
        buf.write(s + '\n')
        try:
            print(s)
        except UnicodeEncodeError:
            print(s.encode('ascii', 'replace').decode())

    hdr = single_series_by_accession()
    registered = series_by_accession()
    sib_log = []
    adj = load_adjudications()
    need = {sib for rules in adj.values() for a, sib, _ in rules if a == 'dup'}
    # the adjudicated siblings' own sentences, per filing, for the 'dup' action
    sib_text = collections.defaultdict(set)
    for r in shards(IN_DIR):
        s_ = (r.get('series_id') or '').strip()
        if s_ in need:
            sib_text[(r.get('accession'), s_)] |= {norm(x) for x in
                                                   split_sentences(r.get('strategy') or '')}
    n_rows = n_cand = 0
    n_recovered = n_unresolved = 0
    lens, ai_sents = [], []
    seen = set()
    with tp.gzip_text_writer(DEST) as out:
        for r in shards(IN_DIR):
            n_rows += 1
            if not r.get('ai_strategy'):
                continue
            sid = (r.get('series_id') or '').strip()
            sname = r.get('series_name')
            resolved = bool(sid)
            if not resolved:
                rec = hdr.get(r.get('accession'))
                if rec:
                    sid, sname = rec[0], rec[1]
                    resolved = True
                    n_recovered += 1
                else:
                    n_unresolved += 1
            key = (r.get('accession'), sid or '__unresolved__')
            if key in seen:
                continue
            seen.add(key)
            sibs = [s for s in registered.get(r.get('accession'), []) if s != sname] \
                if resolved and sname else []
            dropped = []
            extra = None
            if sid in adj:
                dups = set()
                pats = []
                for a, sib, pat in adj[sid]:
                    if a == 'dup':
                        dups |= sib_text.get((r.get('accession'), sib), set())
                    elif a == 'pattern' and pat:
                        pats.append(pat)

                def extra(s, dups=dups, pats=pats):
                    return norm(s) in dups or any(p_ in s.lower() for p_ in pats)
            block, n_ai, n_sent = build_block(r.get('strategy'), sname, sibs if (sibs or extra is None) else ['-'],
                                              dropped, extra)
            if dropped:
                sib_log.append((r.get('accession'), sid, sname, len(dropped), n_ai,
                                dropped[0][:160]))
            if not block:
                continue
            n_cand += 1
            lens.append(len(block))
            ai_sents.append(n_ai)
            out.write(json.dumps({
                'accession': r.get('accession'), 'series_id': sid or None,
                'series_resolved': resolved,
                'series_name': sname, 'cik': r.get('cik'),
                'company': r.get('company'), 'date': r.get('date'),
                'tickers': r.get('tickers'),
                'n_ai_sentences': n_ai, 'n_sentences': n_sent,
                'strategy_chars': len(r.get('strategy') or ''),
                'evidence': block,
                'evidence_sha256': hashlib.sha256(block.encode('utf-8')).hexdigest(),
            }, ensure_ascii=False) + '\n')

    import statistics as st
    w('=' * 96)
    w('FROZEN EVIDENCE BLOCKS')
    w('=' * 96)
    w('  fund-filing rows read        : %d' % n_rows)
    w('  with an AI mention in strategy, block built : %d' % n_cand)
    w('  series_id recovered from the filing header  : %d' % n_recovered)
    w('  series_id UNRESOLVED (multi-series filing)  : %d  <- kept, flagged, and'
      % n_unresolved)
    w('                                                    excluded from fund-level')
    w('                                                    aggregation by name')
    w()
    w('  evidence block length, characters')
    w('    median %6d   mean %6d   p90 %6d   max %6d'
      % (st.median(lens), st.mean(lens), sorted(lens)[int(.9 * len(lens))], max(lens)))
    w('  AI-bearing sentences per filing')
    w('    median %6d   mean %6.1f   p90 %6d   max %6d'
      % (st.median(ai_sents), st.mean(ai_sents),
         sorted(ai_sents)[int(.9 * len(ai_sents))], max(ai_sents)))
    w()
    w('  AI sentences removed as a SIBLING series\' text (sibling.py): %d fund-filings, %d'
      ' of them left with no AI sentence at all'
      % (len(sib_log), sum(1 for x in sib_log if x[4] == 0)))
    for acc, sid_, nm, nd, nk, ex in sib_log:
        w('    %s %s %s: %d removed, %d kept' % (acc, sid_, nm, nd, nk))
        w('        e.g. %s' % ex)
    w()
    w('  For comparison, payloads built from character windows:')
    w('    first classifier  +/-1,200 chars around every AI token, capped')
    w('    fund panel        +/-1,500 chars, head 2,000, capped at 24,000 (mean strategy 22,200)')
    w('  so the block is roughly an order of magnitude smaller and, unlike them, does')
    w('  not move when unrelated text is inserted earlier in the document.')
    w()
    w('wrote %s' % DEST)

    with io.open(os.path.join(LOGS, 's01_extract_evidence.txt'), 'w',
                 encoding='utf-8') as fh:
        fh.write(buf.getvalue())


if __name__ == '__main__':
    main()

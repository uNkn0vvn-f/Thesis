"""Place every held firm on the AI value chain, with an intensity, not a yes/no.

The taxonomy is MULTI-LABEL by design - Global X's own prospectus describes firms
that "develop and utilise" AI and "provide hardware which facilitates" it in one sentence
- so a firm gets a level on each of the four positions rather than being forced into one.
And the level is ORDINAL, because "Microsoft is an AI adopter" and "SoundHound is an AI
company" are not the same statement and a flag cannot tell them apart:

    0  not at all
    1  incidental - present, but a minor part of what the firm does
    2  material - a real and sizeable part, not the core
    3  core - this is what the firm is

    developer    BUILD   - makes AI itself: models, algorithms, AI software
    enabler      SUPPLY  - furnishes what AI runs on: chips, servers, networking, data
                           centres, and the power and cooling those need
    adopter      USE     - puts AI to work inside a business it already has

THERE WERE FOUR. `beneficiary` was dropped after the corpus analysis showed
it was not a position at all. Read against 641 funds' own prospectus language, BENEFIT is
never a sibling of the other three - it is the genus they divide: "the Manager determines
whether it believes such companies BENEFIT from AI by classifying them into two categories:
AI Enablers and AI Adopters". Of the 21 funds whose text defines a benefit-flavoured
category without saying how, not one describes a firm that neither builds, supplies nor
uses. Scored as a fourth sibling it did what a parent must do among its children: it
correlated 0.60 with `enabler`, 93% of the firms it ranked highest also scored enabler >= 2,
and on the day AI got cheaper it priced like a seller rather than like something that gains.

THE QUANTITATIVE ANCHOR IS A SALES SHARE, and that choice is not arbitrary. The fund-claim classifier found
217 filings whose stated inclusion test is a REVENUE THRESHOLD - "at least 50% of revenue
from artificial intelligence" - so the firm label is put in the units the funds themselves
chose to be judged in.

It covers BUILD and SUPPLY only, and says so. The three positions are not commensurable:
building and supplying are things a company SELLS, while using AI is something it DOES, and
using AI earns no revenue - it lowers costs. The data says the same. Firms whose only
position is USE score 1.1% on a revenue share that counts use, against 48.7% for pure
builders, and the correlation between that share and `adopter` is 0.085. A scale that tried to
carry both would score every pure adopter at zero and delete the dimension.

TWO RUNS, TWO DIFFERENT PROMPTS, as for the fund claims. RUN=1 uses PROMPT_A, RUN=2 PROMPT_B. They
differ in phrasing, field order and framing; the CRITERION block between them is byte
identical, because what has to be independent is the two MEASUREMENTS, not the two
definitions. Two definitions of one thing is measurement error wearing the costume of
independence.

NO SELF-REPORT. Asking the model whether it recognises a firm is worthless: a model
claiming to recognise a company is a model being confident, not a model being right.
Instead:

  * `sales_basis` - the sentence from the supplied filing that supports the sales figure,
    quoted verbatim, so the quote can be string-matched against the input. A quote that is
    there means the figure has a checkable basis; a quote that is not means the model
    composed it. Piloted on 300 firms: zero composed quotes, and every firm with a positive
    sales share produced a verifiable one.
  * `business` - a one-line description in the model's own words, so a hallucinated company
    can be caught by reading rather than inferred from a number. To be checked, not
    aggregated.

A firm with no filing text cannot produce a quote, so "none" marks a memory-only label as a
fact about the input rather than as the model's opinion of itself.

A rule that matters sits BESIDE the question that uses it, not at the top of the prompt.
The fund-claim prompts showed this: declaring the NLP rule at the top of a long prompt made
agreement WORSE (73.0% -> 55.0%), and moving it inside the relevant item gave 98.0%.
"""
import io, json, os, threading, time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from src import paths as tp  # noqa: E402

# CLS_OUT / EVID may be overridden so the same classifier can be run on a different
# evidence set - for example, pre-shock 10-Ks - without touching these outputs.
OUT = os.environ.get('CLS_OUT') or tp.out('firm_labels')
LOGS = tp.logs('firm_labels')
os.makedirs(OUT, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)

SRC = tp.out('firm_labels', 'firm_universe.parquet')
EVID = os.environ.get('EVID') or tp.out('firm_labels')
# FIN_ASOF replaces the universe's revenue and market capitalisation - which are the
# LATEST fiscal year available, i.e. 2024-25 - with the last figures that were public
# at a cutoff. Without it a 'pre-shock' prompt tells the model Nvidia is worth $4.6tn
# when the answer it is being asked for is what Nvidia was in 2022 ($614bn).
FIN_ASOF = os.environ.get('FIN_ASOF')
# EVIDENCE=1 attaches the firm's own 10-K Item 1 and its patent CPC profile to the
# prompt. EVIDENCE=0 reproduces the recall-only arm. The prompts are IDENTICAL between
# the two, so their comparison isolates the input and nothing else.
EVIDENCE = os.environ.get('EVIDENCE', '0') == '1'
RUN = os.environ.get('RUN', '1')
KEY = tp.api_key()
URL = 'https://api.deepseek.com/chat/completions'
MODEL = os.environ.get('DS_MODEL', 'deepseek-flash')
WORKERS = int(os.environ.get('WORKERS', '8'))
THINKING = os.environ.get('THINKING', 'off').lower()
LIMIT = int(os.environ.get('LIMIT', '0'))

LEVELS = ('developer', 'enabler', 'adopter')

# The criterion is embedded BYTE-IDENTICALLY in both wordings here and in the fund-side
# classifier. That is the point of it: the diagonal test compares what a fund says it holds
# against what its holdings are, and the two sides have to be measured with one yardstick or
# the comparison is not about the same object. The earlier beneficiary measure showed
# what happens otherwise - the fund side asked about firms that benefit from AI spreading
# while the firm side scored supply-chain adjacency, so the beneficiary column could never
# have landed whatever funds actually held.
#
# The independence that the design needs is independence of EVIDENCE - the fund side reads a
# prospectus written by the adviser, the firm side reads an annual report written by the
# company - and that is untouched. Two different DEFINITIONS are not independence; they are
# measurement error.
#
# The wording is taken from what funds themselves write, so that it can be checked against
# the corpus rather than taken on trust.
CRITERION = """A company can stand in only three relations to artificial intelligence. They are INDEPENDENT - a company may occupy several at once.

"developer" - BUILD. The company makes AI itself: models, algorithms, machine-learning software, AI research. AI is its output. Reselling an AI product somebody else built does not count. Owning data does not count unless the company trains its own models on it.

"enabler" - SUPPLY. The company furnishes the physical layer AI runs on: semiconductors, chip fabrication and its equipment, servers, storage, networking, optical components, data centres, and the electricity, grid hardware and cooling those data centres require - or sells components, materials or specialist services into that layer. Judge what this company sells, not what its customers do with it.

"adopter" - USE. The company puts AI to work inside a business it already has: improving its own operations, or embedding AI into products that are not themselves AI products. AI improves what it already sells rather than being the thing it sells.

Score each on one scale:
  0 = not at all
  1 = mentioned, but incidental
  2 = the company's own annual report describes this as part of its business
  3 = this is what the company is

Level 2 is the line: a fund saying "we invest in companies that build / supply / use AI" means companies at level 2 or above. Judge by how the company describes itself, not by how important the activity seems to you, and not by whether the industry could in principle be affected by AI."""

PROMPT_A = """You are classifying a public company's relationship to artificial intelligence, for academic research on mutual fund holdings.

Answer ONLY with a JSON object. No prose.

""" + CRITERION + """

Judge only from the material below. Where it is silent, score 0 rather than guessing from the industry. Where a business description from the company's own annual report is supplied, answer from it rather than from memory; where a patent profile is supplied, weigh its COMPOSITION - the subclass a company files most patents in is what it mainly does.

Then answer two further questions about SALES.

"ai_sales_share": integer 0-100. Of this company's total revenue, what percentage comes from SELLING into the AI value chain - the BUILD and SUPPLY activity defined above?

The test is WHAT THE BUYER IS PAYING FOR.
  Counts: the customer is buying AI itself, or is buying hardware, components, materials or capacity BECAUSE AI runs on it. An AI software platform. An AI accelerator. Memory or storage sold into AI data centres. The power and cooling those data centres draw.
  Does not count: the customer is buying something else that happens to have AI inside it. A security product, an advertising network or a consulting service that uses machine learning to work better is still selling security, advertising or consulting.
  Does not count: AI used purely in the company's own operations. That lowers costs; it earns no revenue.

Estimate the AI-ATTRIBUTABLE PORTION. This is a share, not a yes-or-no. For a cloud provider, the AI part of cloud, not all of cloud. For a chipmaker serving several end markets, the part going to AI. 0 and 100 should both be rare: 0 means it sells nothing into the chain at all, and 100 means essentially every dollar it earns is AI.

Keep it consistent with the scores you just gave. A company at 3 on BUILD or SUPPLY is usually well above 20. A company at 0 on both should be 0.
Answer -1 only if the material gives you no basis at all for an estimate.

"sales_basis": the single sentence from the material above that best supports your ai_sales_share, quoted VERBATIM and copied exactly. If nothing in the material supports the figure, write exactly "none". Do not paraphrase and do not compose a sentence; it must be copyable text that appears above.

Also return:
"business": one short sentence on what the company does, in your own words; "unknown" if the material does not make it clear.

Return exactly: {"business":"...","developer":0-3,"enabler":0-3,"adopter":0-3,"ai_sales_share":-1 to 100,"sales_basis":"..."}

The company:
"""

PROMPT_B = """Task: for the company described below, judge how it relates to artificial intelligence. This supports academic research into what AI-themed funds actually hold.

Reply with a single JSON object and nothing else.

""" + CRITERION + """

Work from the material supplied below and nothing else. Where it says nothing on a point, the score is 0 - do not fill the gap from what the industry usually does. When the company's own annual-report business section appears below, answer from it and not from memory; when a patent profile appears, read its COMPOSITION, since the subclass a company patents in most is the business it is mainly in.

Two more fields concern SALES rather than position.

"ai_sales_share" - an integer from 0 to 100. What share of this company's revenue is earned by SELLING into the AI value chain, meaning the BUILD and SUPPLY activity described above?

Decide it by asking WHAT THE BUYER IS PAYING FOR.
  It counts when the customer is buying AI itself, or is buying hardware, components, materials or capacity BECAUSE AI runs on it - an AI software platform, an AI accelerator, memory or storage going into AI data centres, the power and cooling those data centres draw.
  It does not count when the customer is buying something else that happens to have AI inside it. A security product, an advertising network or a consulting service that uses machine learning to work better is still selling security, advertising or consulting.
  It does not count when AI is used purely in the company's own operations. That lowers costs; it earns no revenue.

Give the AI-ATTRIBUTABLE PORTION, which is a share and not a verdict. For a cloud provider, the AI part of cloud rather than all of cloud. For a chipmaker with several end markets, the part going to AI. Both 0 and 100 ought to be uncommon: 0 says it sells nothing into the chain at all, 100 says essentially every dollar is AI.
Stay consistent with the scores you have just given: a company at 3 on BUILD or SUPPLY usually sits well above 20, and a company at 0 on both belongs at 0.
Write -1 only when the material offers no basis whatever for an estimate.

"sales_basis" - the one sentence in the material above that best supports the share you gave, reproduced VERBATIM. Write exactly "none" if the material contains nothing that supports it. Do not paraphrase it and do not assemble one; it has to be text that can be copied from above.

"business" - one sentence on what the company does, in your own words; write "unknown" when the material does not make it clear.

Schema: {"developer":0-3,"enabler":0-3,"adopter":0-3,"ai_sales_share":-1 to 100,"sales_basis":"...","business":"..."}

Company details:
"""

ACTIVE = PROMPT_A if RUN in ('1', '3') else PROMPT_B
usage = {'in': 0, 'out': 0, 'calls': 0}
lock = threading.Lock()


BIZ, PROF, FIN = {}, {}, {}
if FIN_ASOF:
    _f = pd.read_parquet(FIN_ASOF)
    for _r in _f.itertuples():
        FIN[str(_r.gvkey)] = (getattr(_r, 'revenue', None), getattr(_r, 'mktcap', None),
                              getattr(_r, 'fin_fyear', None))
    print('as-of financials: %d firms from %s' % (len(FIN), FIN_ASOF))
if EVIDENCE:
    import json as _json
    try:
        with io.open(os.path.join(EVID, 'business_text.jsonl'), encoding='utf-8') as fh:
            for line in fh:
                try:
                    x = _json.loads(line)
                except Exception:
                    continue
                if x.get('business_text'):
                    BIZ[str(x['gvkey'])] = x['business_text']
    except FileNotFoundError:
        pass
    try:
        _p = pd.read_parquet(os.path.join(EVID, 'patent_profile.parquet'))
        PROF = dict(zip(_p.gvkey.astype(str), _p.cpc_profile))
    except FileNotFoundError:
        pass
    print('evidence: %d business descriptions, %d patent profiles' % (len(BIZ), len(PROF)))


def describe(r):
    """Everything we actually have about the firm, and nothing invented."""
    bits = ['name: %s' % r['name']]
    if pd.notna(r.get('ctry')):
        bits.append('country of incorporation: %s' % r['ctry'])
    if pd.notna(r.get('sic4')):
        bits.append('SIC industry code: %d' % int(r['sic4']))
    if pd.notna(r.get('sector')):
        bits.append('broad sector: %s' % r['sector'])
    rev, mcap, fy = r.get('revenue'), r.get('mktcap'), None
    if FIN_ASOF:
        # as-of figures, and SILENCE where none was public at the cutoff: a missing
        # number is the honest input, a current one is a leak.
        rev, mcap, fy = FIN.get(str(r['gvkey']), (None, None, None))
    if rev is not None and pd.notna(rev) and rev > 0:
        bits.append('annual revenue: about %.0f million USD%s'
                    % (rev, ' (fiscal %d)' % fy if fy and pd.notna(fy) else ''))
    if mcap is not None and pd.notna(mcap) and mcap > 0:
        bits.append('market capitalisation: about %.0f million USD' % mcap)
    if EVIDENCE:
        gv = str(r['gvkey'])
        if gv in PROF:
            bits.append('\npatents filed since 2010, by primary CPC subclass '
                        '(read the composition):\n' + str(PROF[gv]))
        if gv in BIZ:
            bits.append('\nthe company describing its own business, from Item 1 of its '
                        'most recent Form 10-K:\n' + str(BIZ[gv])[:9000])
    return '\n'.join(bits)


def lev(v):
    try:
        n = int(float(v))
    except Exception:
        return None
    return min(max(n, 0), 3)


def call(rec, tries=4):
    budgets = (2000, 4000, 6000, 6000)
    # the evidence arm sends far more input; output size is unchanged
    m = {'gvkey': rec['gvkey'], 'name': rec['name'], 'run': RUN,
         'thinking': THINKING}
    for a in range(tries):
        body = {'model': MODEL, 'temperature': 0, 'max_tokens': budgets[min(a, 3)],
                'response_format': {'type': 'json_object'},
                'messages': [{'role': 'user',
                              'content': ACTIVE + describe(rec)}]}
        if THINKING == 'off':
            body['thinking'] = {'type': 'disabled'}
        try:
            r = requests.post(URL, json=body, timeout=180,
                              headers={'Authorization': 'Bearer ' + KEY})
            if r.status_code != 200:
                if r.status_code in (429, 500, 502, 503) and a < tries - 1:
                    time.sleep(3 * (a + 1))
                    continue
                return {**m, 'error': 'HTTP %d: %s' % (r.status_code, r.text[:120])}
            j = r.json()
            with lock:
                usage['in'] += j['usage']['prompt_tokens']
                usage['out'] += j['usage']['completion_tokens']
                usage['calls'] += 1
            ch = j['choices'][0]
            content = (ch['message'].get('content') or '').strip()
            if not content:
                if a < tries - 1:
                    continue
                return {**m, 'error': 'empty content, finish_reason=%s'
                        % ch.get('finish_reason')}
            try:
                p = json.loads(content)
            except json.JSONDecodeError:
                p, _ = json.JSONDecoder().raw_decode(content[content.index('{'):])
            res = {**m, 'error': None}
            res['business'] = str(p.get('business', ''))[:300]
            for f in LEVELS:
                res[f] = lev(p.get(f))
            try:
                res['ai_sales_share'] = float(p.get('ai_sales_share'))
            except Exception:
                res['ai_sales_share'] = None
            # Kept whole and unclipped: it is checked by string match against the prompt
            # input, so truncating it here would manufacture failures. Unlike a self-report
            # of whether the model knows the firm, this is checkable by someone else.
            res['sales_basis'] = str(p.get('sales_basis', ''))
            return res
        except Exception as e:
            if a < tries - 1:
                time.sleep(3 * (a + 1))
                continue
            return {**m, 'error': repr(e)[:150]}


def main():
    F = pd.read_parquet(SRC)
    if LIMIT:
        F = F.head(LIMIT)
    recs = F.to_dict('records')

    dest = os.path.join(OUT, 'labels_%srun%s.jsonl'
                        % ('ev_' if EVIDENCE else '', RUN))
    # Only a row that ANSWERED counts as done. Counting every gvkey found, error rows
    # included, would mean a run interrupted by the API - a 402 on an empty balance, a
    # rate limit, a network drop - could not be resumed: the retry would see the failures
    # as finished work and write a label file that was mostly errors, with nothing
    # downstream any the wiser. Errored rows are left in place as a record and
    # simply re-attempted; `keep` drops the stale line so the file does not grow a second
    # entry for the same firm.
    done, errored = set(), 0
    if os.path.exists(dest):
        keep = []
        with io.open(dest, encoding='utf-8') as fh:
            for line in fh:
                try:
                    x = json.loads(line)
                except Exception:
                    continue
                if x.get('error'):
                    errored += 1
                    continue
                done.add(x['gvkey'])
                keep.append(line)
        if errored:
            with io.open(dest, 'w', encoding='utf-8') as fh:
                fh.writelines(keep)
    todo = [r for r in recs if r['gvkey'] not in done]
    if errored:
        print('  %d row(s) from an earlier run carried an error and will be retried'
              % errored)
    print('run %s | prompt %s | evidence %s | model %s | thinking %s'
          % (RUN, 'A' if RUN in ('1', '3') else 'B',
             'ON' if EVIDENCE else 'off', MODEL, THINKING))
    print('firms %d, already done %d, to do %d' % (len(recs), len(done), len(todo)))
    tp.no_api_stop(len(todo), 'firms')

    t0 = time.time()
    n = 0
    with io.open(dest, 'a', encoding='utf-8') as fh, \
            ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for res in ex.map(call, todo):
            fh.write(json.dumps(res, ensure_ascii=False) + '\n')
            n += 1
            if n % 200 == 0:
                fh.flush()
                el = time.time() - t0
                print('  %5d / %5d   %.0f s   %.1f firms/s   in %d out %d tok'
                      % (n, len(todo), el, n / max(el, 1), usage['in'], usage['out']))
    print('done %d in %.0f s | calls %d | tokens in %d out %d'
          % (n, time.time() - t0, usage['calls'], usage['in'], usage['out']))


if __name__ == '__main__':
    main()

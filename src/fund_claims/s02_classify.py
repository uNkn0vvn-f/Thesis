"""Classification: classify the frozen evidence block with independent questions.

Nothing here asks the model to pick one label from a ranked list. A prompt with a
"strict priority order from AI-Autonomous > AI-Augmented >
AI Support > AI-Theme Assets" is where the instability lives: the three categories inside
that ordering are the three least reproducible (63%, 71%, 87% against 93% for AI-Theme
Assets), and a fund that both uses AI and invests in AI companies is silently recorded as
only the first.

So every question below is independent and may be answered yes simultaneously:

  A  uses_ai            does the fund say it uses AI in its OWN investment process?
  B  invests_ai_theme   does the fund say it invests in AI-related COMPANIES?
  C  three value-chain labels plus two ways of naming nothing, multi-label, because the
     documents name several at once - Global X: "development and utilization ...
     positioned to benefit ... provide hardware which facilitates" is developer + adopter
     + enabler in one sentence, and forcing a single choice there is what produced the
     flips. The three positions are defined by a CRITERION block held byte-identical with
     the firm-side classifier, so that both sides of the diagonal test mean one thing by
     each word; `chain_benefit_vague` replaced `chain_beneficiary` once the corpus showed
     BENEFIT is the genus the three divide rather than a fourth sibling, and it is scored
     but never compared against a company, because it names no position at all.
  D  four inclusion-test labels, also multi-label; the single "strictest test stated"
     axis is derived deterministically in s03, not chosen by the model
  E  risk_only          fallback when neither A nor B

The category is composed by rule in s03. The model never sees a category name.

Run twice with RUN=1 and RUN=2 - these are the two independent measurements that
can be used as instruments for each other in measurement-error analysis.
"""
import gzip, io, json, os, random, threading, time
from concurrent.futures import ThreadPoolExecutor

import requests

from src import paths as tp  # noqa: E402

OUT = tp.out('fund_claims')
os.makedirs(OUT, exist_ok=True)

SRC = os.path.join(OUT, 'evidence.jsonl.gz')
RUN = os.environ.get('RUN', '1')
KEY = tp.api_key()
URL = 'https://api.deepseek.com/chat/completions'
# The endpoint serves exactly two models - deepseek-flash and deepseek-v4-pro. The
# name "deepseek-v4-flash" used by the first classifier (disclosure) is an alias that
# resolves to deepseek-flash, so use the canonical name.
MODEL = os.environ.get('DS_MODEL', 'deepseek-flash')
WORKERS = int(os.environ.get('WORKERS', '8'))
LIMIT = int(os.environ.get('LIMIT', '0'))
# deepseek-flash is a reasoning model and bills the sampled chain of thought as output
# tokens: roughly 2,400 per call here, which is where nearly all the spend goes.
# thinking={"type":"disabled"} suppresses it. Of the documented ways to ask, only this
# one has any effect - reasoning_effort and enable_thinking are silently ignored.
# Default off. Measured on 80 filings, twice each: suppressing the chain of thought
# cuts output tokens 17.6x (309,200 -> 17,553) and wall time 5.6x, and the labels come
# back MORE reproducible, not less - mean flag agreement between two identical calls
# rises from 97.3% to 98.2% and the share of filings where all fifteen flags repeat
# exactly from 72.5% to 82.3%. That is what the initial diagnostic predicted: the sampled
# reasoning path was itself the source of the non-determinism.
# Caveat carried into the README: this shows greater REPRODUCIBILITY, not greater
# accuracy. The two settings agree on 96.4% of flags; which is right where they differ
# needs the hand-coded subsample.
THINKING = os.environ.get('THINKING', 'off').lower()
SUFFIX = os.environ.get('SUFFIX', '')
DEST = os.path.join(OUT, 'labels_run%s%s.jsonl' % (RUN, SUFFIX))

# Keep this BYTE-IDENTICAL to CRITERION in src/firm_labels/c01_classify.py.
#
# That identity is the point. The diagonal test asks whether a fund that SAYS it holds
# builders actually holds builders, and it can only ask that if both sides mean the same
# thing by "builder". The earlier beneficiary measure showed what happens when they do
# not: this side asked about companies that benefit from AI spreading WITHOUT supplying it,
# the firm side scored supply-chain adjacency, and the beneficiary column was therefore
# incapable of landing whatever funds actually held. The two sides were never testing the
# same object.
#
# What the design needs to keep independent is the EVIDENCE - a prospectus written by the
# adviser here, an annual report written by the company there - and that is untouched. Two
# different definitions of one word are not independence; they are measurement error.
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

PROMPT = """You are reading the "Principal Investment Strategies" text of a US mutual fund or ETF, taken from its SEC Form 485BPOS prospectus. Only the sentences mentioning artificial intelligence, plus their immediate context and the fund's 80%-of-assets policy sentence, are shown.

Answer each question below INDEPENDENTLY. Several may be "Y" at the same time. There is no ranking and no priority order: do not suppress one answer because another is "Y".

WHAT COUNTS AS AI OR MACHINE LEARNING, throughout every question below:
artificial intelligence; machine learning; deep learning; neural networks; NATURAL LANGUAGE PROCESSING (NLP); large language models; generative AI; computer vision; reinforcement learning; genetic or evolutionary algorithms. Natural language processing counts as AI here even where the text never uses the words "artificial intelligence" or "machine learning".
Ordinary quantitative methods do NOT count on their own: statistical models, factor models, optimisation, rules-based screens and algorithmic execution are not AI unless the text ties them to one of the techniques named above.

--------------------------------------------------------------------
A. uses_ai
Does the text affirmatively state that AI or machine learning helps decide what this fund holds - selecting, weighting, ranking, screening or timing its securities?

For THIS question, "AI or machine learning" includes natural language processing (NLP) - a fund that says it uses NLP to read filings, news or transcripts and then acts on that in its security selection is using AI, even where the text never writes "artificial intelligence" or "machine learning".

Answer Y regardless of WHO operates that AI. It counts whether the operator is the fund's adviser, its sub-adviser, the methodology of an index the fund tracks, a model provider, or a third-party classification system. Who operates it is recorded separately in the next question and must not affect this one.
Answer Y only if the use is affirmative. Permissive wording alone ("may use", "could use", "from time to time", "on a limited basis") is NOT sufficient. Standard forward-looking drafting about an existing tool ("is expected to be used", "will continue to use") DOES count.
Answer N if the AI belongs only to the companies the fund invests in, or if AI is used only for research support, operations, compliance or client service and not for deciding holdings.

ai_operator_third_party
Is the AI MODEL OR METHODOLOGY that helps decide the fund's holdings supplied by someone who is not an adviser to the fund - for instance an index provider whose published methodology applies AI, or a model licensed from a firm with no advisory role?

The test is who owns the method, not whose computers it runs on. Answer N when the adviser or sub-adviser built or owns the model, even if it runs on third-party computing infrastructure, a commercial AI platform, or third-party data. Answer N when uses_ai is N.

uses_ai_hedged
Does the text say the fund, its adviser or sub-adviser MAY use AI or machine learning in its investment process, without stating that it actually does? Answer Y for permissive wording only - "may use", "may utilize", "could use", "from time to time", "on a limited basis", "where appropriate". If the text states the use affirmatively, uses_ai is Y and this one is N.

uses_ai_support
Does the text say the fund, its adviser or sub-adviser uses AI or machine learning for work that is NOT choosing what the fund holds - reading, summarising, extracting or organising information, generating reports, compliance monitoring, back-office operations, trade execution mechanics, or client service? Answer Y whether that use is affirmative or hedged, and independently of the two questions above.

B. invests_ai_theme
Does the text connect what the fund HOLDS to artificial intelligence in any way at all?

Answer Y broadly. It is enough that the text ties the fund's holdings to AI - as a theme, a sector, an index, a selection criterion, or merely by describing a company the fund holds as an AI business. Do NOT answer N because the connection seems incidental, because AI is one item in a longer list of technologies, or because the fund is written on a single stock. Whether AI is the fund's own basis for choosing, or only a fact about something it happens to own, is recorded separately in the next two questions and must not affect this one.

ai_selection_criterion
Does the text make artificial intelligence part of the fund's OWN basis for choosing what it holds - the theme it pursues, the index it tracks, the sector it targets, or a condition a company has to meet to be eligible for the portfolio?

ai_only_in_holding_description
Is every mention of artificial intelligence confined to describing what a particular company or underlying security does, rather than to how the fund picks? This is the usual pattern in single-stock and leveraged products, where the prospectus describes the one company the fund is written on. Answer N if any AI mention forms part of the fund's own selection basis.

--------------------------------------------------------------------
C. If B is Y, which kinds of AI-related company does the text name? Answer each one separately; a fund that names several gets several Y.

""" + CRITERION + """

The 0-3 scale above is how the COMPANIES are scored elsewhere in this research, against their own annual reports. Here you are reading a fund's prospectus, so you only record whether the text names the position at all.

Some of the tests above are about the COMPANY and are answerable only from a company's own filing - whether it built the AI it sells rather than reselling someone else's, whether it trains models on the data it owns. A prospectus almost never addresses them. Where this text does not, decide on what it DOES say; do not treat its silence as a failure to qualify. The question is whether the fund names the position, not whether you can verify the company clears every part of the definition.

chain_developer     the text says the fund invests in companies that BUILD, as defined above.
chain_enabler       the text says the fund invests in companies that SUPPLY, as defined above.
chain_adopter       the text says the fund invests in companies that USE, as defined above.

Two ways of naming nothing. They are NOT positions and are never compared against a company's score; they record how the claim was made.

chain_unspecified    the text says the fund invests in AI-related companies but does NOT say what those companies actually do with AI. It names AI as a theme, a sector, an index or one item in a list of technologies, and leaves the relationship unstated.
chain_benefit_vague  the text says the companies BENEFIT or "stand to benefit", "are positioned to benefit", "have the potential to benefit" from the growth of AI, and does not say whether they build, supply or use it.

You are recording what the text SAYS, not judging what the fund probably means. Do not infer a position the text does not state. If the text only names the theme - "companies involved in artificial intelligence", "the artificial intelligence sector", "artificial intelligence, robotics, cybersecurity and virtual reality" - then chain_unspecified is Y and the three positions are N.

If B is N, answer N to all five.

--------------------------------------------------------------------
D. If B is Y, what test does the text give for deciding whether a company counts as AI-related? Answer each separately; a fund may state more than one.

test_revenue        a quantitative threshold on the company's own business - a percentage of revenue, earnings, assets or operations derived from AI-related activity (e.g. "derives 50% or more of its revenue from").
test_index_provider the definition is delegated to a named third-party index provider or index methodology (e.g. "as defined by Indxx, LLC", "the Index Provider identifies", "the Underlying Index is designed to").
test_manager        the adviser or portfolio managers decide, by their own judgement or belief (e.g. "that the portfolio managers believe", "the Adviser believes", "in the Adviser's view").

If the text states no test at all, answer N to all three; do not invent one. If B is N, answer N to all three.

--------------------------------------------------------------------
E. risk_only
Is EVERY mention of AI in the text confined to risk factors, disclaimers, or statements about what AI might do TO the fund or the market - with no statement that the fund uses AI and no statement that it invests in AI companies?

--------------------------------------------------------------------
Return a json object with exactly these keys, each "Y" or "N" except the quotes:

{"uses_ai": "Y|N",
 "ai_operator_third_party": "Y|N",
 "uses_ai_hedged": "Y|N",
 "uses_ai_support": "Y|N",
 "invests_ai_theme": "Y|N",
 "ai_selection_criterion": "Y|N",
 "ai_only_in_holding_description": "Y|N",
 "chain_developer": "Y|N",
 "chain_enabler": "Y|N",
 "chain_adopter": "Y|N",
 "chain_benefit_vague": "Y|N",
 "chain_unspecified": "Y|N",
 "test_revenue": "Y|N",
 "test_index_provider": "Y|N",
 "test_manager": "Y|N",
 "risk_only": "Y|N",
 "quote_uses": "<one complete sentence copied VERBATIM from the passage supporting uses_ai=Y, else empty string; do not paraphrase, do not join fragments with an ellipsis>",
 "quote_theme": "<one complete sentence copied VERBATIM from the passage supporting invests_ai_theme=Y, else empty string; do not paraphrase, do not join fragments with an ellipsis>",
 "quote_test": "<one complete sentence copied VERBATIM from the passage showing the inclusion test, else empty string; do not paraphrase, do not join fragments with an ellipsis>"}"""

# ---------------------------------------------------------------------------- #
# The second measurement.
#
# Measurement-error analysis treats the two runs as two noisy readings of the same latent
# label. That identification needs their errors to be independent given the truth,
# and two identical calls to one model do not deliver it: a model that misreads a
# particular turn of phrase misreads it both times, so agreement between identical
# runs measures sampling noise and silently reports systematic bias as accuracy.
#
# PROMPT_B asks the same questions in a different order, in different words, with
# different illustrative examples. That decorrelates the part of the error that comes
# from how the question is worded. It does NOT decorrelate the model's own systematic
# bias - only a different model or a human coder would - and 04 says so.
# ---------------------------------------------------------------------------- #
PROMPT_B = """Below is an extract from the "Principal Investment Strategies" section of a US fund's SEC Form 485BPOS prospectus. It contains the sentences that mention artificial intelligence, the sentences immediately around them, and the sentence stating the fund's 80%-of-assets policy.

Your task is to record what the passage says. Record each item on its own merits. Items are not mutually exclusive and there is no ordering among them, so never mark one item "N" merely because another is "Y". Record only what is written; do not supply information the passage omits.

THE SCOPE OF "ARTIFICIAL INTELLIGENCE OR MACHINE LEARNING" for every item below:
it covers artificial intelligence, machine learning, deep learning, neural networks, NATURAL LANGUAGE PROCESSING (NLP), large language models, generative AI, computer vision, reinforcement learning, and genetic or evolutionary algorithms. Treat natural language processing as within scope even when the passage nowhere writes "artificial intelligence" or "machine learning".
Plain quantitative technique on its own is outside scope: statistical or factor models, optimisation, rules-based screening and algorithmic order execution do not qualify unless the passage connects them to one of the techniques just listed.

== Part 1. The portfolio ==

invests_ai_theme
Mark Y whenever the passage links what the fund holds to artificial intelligence, on any footing whatsoever.

Be generous here. A link counts whether it is framed as a theme, an industry, a tracked index, a rule for picking holdings, or nothing more than a remark that some business the fund owns works in AI. Do not withhold Y because the link looks incidental, because AI sits in a long list of technologies alongside others, or because the fund is built on one stock. The question of whose AI it is - the fund's own yardstick for picking, or just a fact about something it owns - belongs to the next two items and must leave this one untouched.

ai_selection_criterion
Mark Y when artificial intelligence enters the fund's own grounds for deciding what to hold: the theme it pursues, the index it follows, the industry it aims at, or a qualification a business must satisfy to be eligible.

ai_only_in_holding_description
Mark Y when every artificial-intelligence reference in the passage describes what some particular business or underlying security does, and none of them describes how the fund chooses. Single-stock and leveraged products usually read this way, since their prospectus profiles the one business the product is written on. Mark N if any AI reference belongs to the fund's own grounds for picking.

== Part 2. What those businesses do, if invests_ai_theme is Y ==

Mark every item the passage actually describes. A passage describing several gets several Y. If invests_ai_theme is N, all five are N.

""" + CRITERION + """

The 0-3 scale just given is how the COMPANIES are scored elsewhere in this research, against their own annual reports. Your passage is a fund's prospectus, so all you record is whether it names the position.

Parts of those definitions concern the COMPANY and can be settled only from a company's own filing - did it build the AI it sells or resell someone else's, does it train models on the data it holds. A prospectus hardly ever speaks to that. When your passage is silent on such a point, go by what it DOES say; silence is not a failure to qualify. You are asked whether the fund names the position, not whether the companies can be shown to satisfy every clause.

chain_developer
The passage says the fund invests in businesses that BUILD, as set out above.

chain_enabler
The passage says the fund invests in businesses that SUPPLY, as set out above.

chain_adopter
The passage says the fund invests in businesses that USE, as set out above.

The next two are ways of naming nothing. They are NOT positions, and are never set against a company's score; they record the manner of the claim.

chain_unspecified
The passage names artificial intelligence as the subject but never says what the businesses do with it. It may list AI as a theme, an industry, or one entry among several technologies, and stop there.

chain_benefit_vague
The passage says the businesses gain, or may come to gain, from artificial intelligence spreading - phrases such as "benefiting from", "positioned to benefit", "have the potential to benefit" - without saying whether they build it, supply it or use it.

Record what the passage SAYS, not what the fund probably means. A passage that says no more than "companies involved in artificial intelligence", "the artificial intelligence industry", or "artificial intelligence, robotics and cybersecurity" is chain_unspecified = Y with the three positions N. Resist the temptation to guess which part of the industry is meant.

== Part 3. How a business qualifies, if invests_ai_theme is Y ==

Mark every item the passage states; more than one may apply. If invests_ai_theme is N, all four are N.

test_revenue
A measurable bar on the business itself - a stated share of its revenue, earnings, assets or operations coming from AI-related activity. Example wording: "at least half of its revenues".

test_index_provider
The passage hands the decision to an outside index or its methodology, naming the provider or the index. Example wording: "as determined by the Index Provider", "companies included in the Underlying Index".

test_manager
The decision rests with the fund's own adviser or portfolio managers and their opinion. Example wording: "which the Adviser considers", "companies the portfolio managers regard as".

Where the passage sets out no bar whatsoever, leave all three at N rather than reaching for one.

== Part 4. The fund's own method ==

uses_ai
Mark Y when the passage states that artificial intelligence or machine learning takes part in settling what this fund holds - choosing securities, setting weights, ranking or screening candidates, or timing trades.

For THIS item, natural language processing (NLP) counts as artificial intelligence or machine learning - a fund that says it applies NLP to filings, news or transcripts and feeds that into picking its holdings is using AI, even where the passage never writes "artificial intelligence" or "machine learning".

Whose hands the AI is in makes no difference here. The adviser, a sub-adviser, the rulebook of an index the fund follows, a model supplier, a licensed platform, an outside classification engine - any of them counts. The identity of the operator is recorded in the item straight after this one and must not sway this one.
Requires a statement that this is actually done. Language leaving it optional - "may employ", "might make use of", "where appropriate", "occasionally" - does not qualify. Ordinary forward-looking phrasing about a tool already in place does qualify.
Mark N when the artificial intelligence belongs solely to the businesses in the portfolio, or when it serves only research assistance, operations, compliance or client servicing and plays no part in settling holdings.

ai_operator_third_party
Mark Y when the artificial-intelligence METHOD bearing on the fund's holdings comes from a party with no advisory role at the fund - an index provider whose published rulebook applies AI, or a model licensed in from such a firm.

What matters is ownership of the method, not the hardware or data behind it. Mark N when the adviser or sub-adviser devised or owns the model, even where it runs on someone else's computing platform, a commercial AI service, or bought-in data. Mark N whenever uses_ai is N.

uses_ai_hedged
Mark Y when the passage keeps the fund's own use of artificial intelligence optional rather than asserting it - "may employ", "may utilise", "might use", "from time to time", "on a limited basis", "where appropriate" - applied to deciding what the fund holds. When the passage asserts the use outright, uses_ai is Y and this one is N.

uses_ai_support
Mark Y when the passage says the fund, its adviser or its sub-adviser turns artificial intelligence or machine learning to work other than picking holdings: digesting, summarising, pulling out or filing information, writing reports, monitoring compliance, running the back office, handling the mechanics of trading, or dealing with clients. Applies whether or not that use is asserted outright, and independently of the two items above.

== Part 5. Fallback ==

risk_only
Mark Y only when every reference to artificial intelligence in the passage sits in a risk factor, a disclaimer, or a statement about how AI might affect the fund or markets, and the passage neither says the fund applies AI nor says it buys AI-related businesses.

== Output ==

Reply with a json object holding exactly these keys, each "Y" or "N" apart from the three quotes:

{"uses_ai": "Y|N",
 "ai_operator_third_party": "Y|N",
 "uses_ai_hedged": "Y|N",
 "uses_ai_support": "Y|N",
 "invests_ai_theme": "Y|N",
 "ai_selection_criterion": "Y|N",
 "ai_only_in_holding_description": "Y|N",
 "chain_developer": "Y|N",
 "chain_enabler": "Y|N",
 "chain_adopter": "Y|N",
 "chain_benefit_vague": "Y|N",
 "chain_unspecified": "Y|N",
 "test_revenue": "Y|N",
 "test_index_provider": "Y|N",
 "test_manager": "Y|N",
 "risk_only": "Y|N",
 "quote_uses": "<a whole sentence copied word for word from the passage supporting uses_ai=Y, otherwise an empty string; do not reword it and do not splice pieces together with an ellipsis>",
 "quote_theme": "<a whole sentence copied word for word from the passage supporting invests_ai_theme=Y, otherwise an empty string; do not reword it and do not splice pieces together with an ellipsis>",
 "quote_test": "<a whole sentence copied word for word from the passage showing the qualifying bar, otherwise an empty string; do not reword it and do not splice pieces together with an ellipsis>"}"""

PROMPTS = {'1': PROMPT, '2': PROMPT_B}
ACTIVE_PROMPT = PROMPTS.get(RUN, PROMPT)
HEADERS = {'1': '\n\n--- FUND STRATEGY TEXT ---\n',
           '2': '\n\n=== PASSAGE ===\n'}
ACTIVE_HEADER = HEADERS.get(RUN, HEADERS['1'])

FLAGS = ('uses_ai', 'ai_operator_third_party', 'uses_ai_hedged',
         'uses_ai_support', 'invests_ai_theme',
         'ai_selection_criterion', 'ai_only_in_holding_description',
         'chain_developer', 'chain_enabler', 'chain_adopter', 'chain_benefit_vague',
         'chain_unspecified', 'test_revenue', 'test_index_provider', 'test_manager',
         'risk_only')
META = ('accession', 'series_id', 'series_name', 'cik', 'company', 'date',
        'evidence_sha256', 'n_ai_sentences')

lock = threading.Lock()
usage = {'in': 0, 'out': 0, 'calls': 0}


def yn(v):
    s = str(v or '').strip().upper()
    return 'Y' if s.startswith('Y') or s == 'TRUE' else 'N'


def call(rec, tries=4):
    # DeepSeek's reasoning model spends max_tokens on its chain of thought first. If the
    # budget runs out, content comes back as an empty string with finish_reason 'length',
    # which a parser would report as malformed JSON. Budget generously and escalate on
    # retry.
    budgets = (8000, 16000, 16000, 16000)
    m = {k: rec.get(k) for k in META}
    m['run'] = RUN
    m['thinking'] = THINKING
    for a in range(tries):
        body = {'model': MODEL, 'temperature': 0, 'max_tokens': budgets[min(a, 3)],
                'response_format': {'type': 'json_object'},
                'messages': [{'role': 'user',
                              'content': ACTIVE_PROMPT + ACTIVE_HEADER
                              + rec['evidence']}]}
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
                    continue          # retry with a bigger budget
                return {**m, 'error': 'empty content, finish_reason=%s'
                        % ch.get('finish_reason')}
            # The model occasionally emits the object twice, back to back, which
            # json.loads rejects as "Extra data". Take the first complete object.
            try:
                p = json.loads(content)
            except json.JSONDecodeError:
                p, _ = json.JSONDecoder().raw_decode(content[content.index('{'):])
            res = {**m, 'error': None}
            for f in FLAGS:
                res[f] = yn(p.get(f))
            for q in ('quote_uses', 'quote_theme', 'quote_test'):
                res[q] = str(p.get(q, ''))[:300]
            return res
        except Exception as e:
            if a < tries - 1:
                time.sleep(3 * (a + 1))
                continue
            return {**m, 'error': repr(e)[:150]}


def main():
    cands = []
    with gzip.open(SRC, 'rt', encoding='utf-8') as fh:
        for line in fh:
            cands.append(json.loads(line))

    done = set()
    if os.path.exists(DEST):
        with io.open(DEST, encoding='utf-8') as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if not r.get('error'):
                    # done means answered for THIS text: a key whose evidence changed
                    # (s00 attribution) is asked again rather than inheriting a verdict
                    done.add((r['accession'], r['series_id'], r.get('evidence_sha256')))
    todo = [c for c in cands
            if (c['accession'], c['series_id'], c.get('evidence_sha256')) not in done]
    if LIMIT:
        todo = todo[:LIMIT]
    # shuffle so a partial run is a random sample rather than the earliest filings
    random.Random(int(RUN)).shuffle(todo)
    print('run=%s  total=%d  done=%d  to call=%d' % (RUN, len(cands), len(done), len(todo)),
          flush=True)
    tp.no_api_stop(len(todo), 'evidence blocks')
    if not todo:
        print('nothing to do')
        return

    t0 = time.time()
    ok = err = 0
    with io.open(DEST, 'a', encoding='utf-8') as fh, \
            ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, res in enumerate(ex.map(call, todo), 1):
            fh.write(json.dumps(res, ensure_ascii=False) + '\n')
            ok += 0 if res.get('error') else 1
            err += 1 if res.get('error') else 0
            if i % 100 == 0:
                fh.flush()
                el = time.time() - t0
                print('  %d/%d  %.1f min  ok=%d err=%d  ETA %.1f min'
                      % (i, len(todo), el / 60, ok, err,
                         (len(todo) - i) * (el / i) / 60), flush=True)
    print('DONE %.1f min ok=%d err=%d calls=%d tokens in=%d out=%d'
          % ((time.time() - t0) / 60, ok, err, usage['calls'], usage['in'], usage['out']))


if __name__ == '__main__':
    main()

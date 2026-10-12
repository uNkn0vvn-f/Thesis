"""Second-model labels: run the classifier prompts with an OpenAI model and compare.

    python src/analysis/robustness/cross_model_labels.py estimate                 # no calls
    python src/analysis/robustness/cross_model_labels.py models --filter luna     # list model ids
    python src/analysis/robustness/cross_model_labels.py run --model <id> [--scope sample|all]
    python src/analysis/robustness/cross_model_labels.py compare --model <id>

The fund prompt (first wording) and the company prompt (first wording, with the same
filing text, patent profile and as-of financials for each annual vintage) are sent
unchanged, so the comparison varies the model and nothing else. Fund inputs are the AI
evidence passages; company inputs are built with src.firm_labels.c01_classify.describe.

Scope `sample` is the stratified validation sample (140 fund filings, 140 company-years);
`all` is every AI-bearing fund filing and every company-year held by the AI-themed sample
funds (DeepSeek theme label, at least half the value classified); `luna-theme` adds the
company-years held by fund-quarters that are AI-themed under the second model's own fund
labels (run after `all`; luna_measures.py builds those labels); `universe` is every
company-year of the classified universe, 2020-2026 (the DeepSeek vintages); `preevent` is
the pre-event company labels of the event analysis, one information set per event (company_preevent.jsonl). `estimate` reports calls
and tokens for the first two scopes without a key.

The API key is read from the environment variable OPENAI_API_KEY, or from the key file
(OPENAI_API_KEY_FILE, default `API Key.txt` beside the project folder), using only a line
that names OpenAI or a key in OpenAI's project format. It is never printed or written.
Responses are cached in data/interim/cross_model/<model>/ and never requested twice; every
request and raw response is logged there (calls.jsonl.gz) with an AI-usage record
(AI_USAGE_RECORD.md) for the thesis AI-usage table.

Agreement between two language models is a consistency check across model families, not
accuracy: both read the same text and can share errors. Human coding remains the benchmark.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('THESIS_NO_API', '1')        # the imported modules must not load a key
from src.fund_claims import s02_classify as S02  # noqa: E402
from src.firm_labels import c01_classify as C01  # noqa: E402
from src.analysis.robustness.common import (TABLES, INTERIM, PROCESSED,  # noqa: E402
                                            load_exposure_table, load_holdings)

API = 'https://api.openai.com/v1'
DISPLAY = {'gpt-6-luna': 'GPT-6 Luna'}     # the model's public name, for prose
VAL = INTERIM / 'validation'
VINT = INTERIM / 'firm_labels' / 'vintages'
FUND_FLAGS = ['invests_ai_theme', 'ai_selection_criterion', 'chain_developer', 'chain_enabler',
              'chain_adopter', 'uses_ai', 'risk_only']
ROLES = ['developer', 'enabler', 'adopter']


# ---------------------------------------------------------------- inputs
def fund_records(scope):
    ev = {}
    with gzip.open(INTERIM / 'fund_claims' / 'evidence.jsonl.gz', 'rt', encoding='utf-8') as fh:
        for line in fh:
            r = json.loads(line)
            ev[(r['accession'], r['series_id'])] = r
    if scope == 'sample':
        K = pd.read_csv(VAL / 'KEY_fund_claims_model_labels_DO_NOT_OPEN_BEFORE_CODING.csv', dtype=str)
        keys = list(zip(K.accession, K.series_id))
    else:
        keys = list(ev)
    out = []
    for k in keys:
        r = ev.get(k)
        if r:
            out.append({'key': '%s|%s' % k, 'accession': k[0], 'series_id': k[1],
                        'evidence_sha256': r.get('evidence_sha256'),
                        'content': S02.PROMPTS['1'] + S02.HEADERS['1'] + r['evidence']})
    return out


_VINTAGE = {}


def set_vintage(y):
    """Point the company classifier's module state at one annual evidence set."""
    if y not in _VINTAGE:
        _VINTAGE[y] = load_evidence(VINT / '00_evidence' / str(y))
    C01.BIZ, C01.PROF, C01.FIN = _VINTAGE[y]
    C01.EVIDENCE = True
    C01.FIN_ASOF = 'as-of'


PRE = INTERIM / 'analysis' / 'preevent'


def preevent_records():
    """The pre-event company labels: one information set per event (filings before
    the event, patents filed up to its year, financials public at the cutoff), the companies
    and descriptions the DeepSeek pre-event labels were built from (built by
    src.firm_labels.run_preevent into data/interim/analysis/preevent/)."""
    U = pd.read_parquet(PRE / 'firm_universe_preevent.parquet')
    U['gvkey'] = U.gvkey.astype(str)
    U = U.set_index('gvkey')
    out = []
    for tag in ('chatgpt', 'deepseek'):
        C01.BIZ, C01.PROF, C01.FIN = load_evidence(PRE / '00_evidence' / tag)
        C01.EVIDENCE = True
        C01.FIN_ASOF = 'as-of'
        L = pd.read_parquet(PRE / '02_label' / 'out' / tag / 'firm_ai_labels.parquet', columns=['gvkey'])
        for g in sorted(L.gvkey.astype(str)):
            rec = U.loc[g].to_dict()
            rec['gvkey'] = g
            out.append({'key': '%s|%s' % (g, tag), 'gvkey': g, 'event': tag,
                        'content': C01.PROMPT_A + C01.describe(rec)})
    return out


def load_evidence(d):
    """Business text, patent profile and as-of financials of one evidence directory, read
    as src/firm_labels/c01_classify.py reads them."""
    biz = {}
    p = d / 'business_text.jsonl'
    if p.exists():
        with open(p, encoding='utf-8', errors='replace') as fh:
            for line in fh:
                try:
                    x = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if x.get('business_text'):
                    biz[str(x['gvkey'])] = x['business_text']
    prof = {}
    if (d / 'patent_profile.parquet').exists():
        P = pd.read_parquet(d / 'patent_profile.parquet')
        prof = dict(zip(P.gvkey.astype(str), P.cpc_profile))
    fin = {}
    if (d / 'financials_asof.parquet').exists():
        Fn = pd.read_parquet(d / 'financials_asof.parquet')
        for r in Fn.itertuples():
            fin[str(r.gvkey)] = (getattr(r, 'revenue', None), getattr(r, 'mktcap', None),
                                 getattr(r, 'fin_fyear', None))
    return biz, prof, fin


def company_records(scope):
    U = pd.read_parquet(INTERIM / 'firm_labels' / 'firm_universe.parquet')
    U['gvkey'] = U.gvkey.astype(str)
    U = U.set_index('gvkey')
    if scope == 'sample':
        K = pd.read_csv(VAL / 'KEY_company_roles_model_labels_DO_NOT_OPEN_BEFORE_CODING.csv', dtype={'gvkey': str})
        pairs = list(zip(K.gvkey, K.yr.astype(int)))
    elif scope == 'luna-theme':
        from src.analysis.robustness import luna_measures as LM
        E = load_exposure_table('deepseek')
        P = LM.build_panel(pd.read_parquet(INTERIM / 'fund_claims' / 'panel_fund_filing.parquet'), LM.fund_labels({}))
        A, _ = LM.asof_claims(E, P)
        D = A[A.theme_broad & (E.set_index(['series_id', 'q']).labelled.reindex(
            pd.MultiIndex.from_frame(A[['series_id', 'q']])).to_numpy() >= 50)][['series_id', 'q']]
        H = load_holdings().merge(D, on=['series_id', 'q'])
        pairs = sorted(set(zip(H.gvkey, H.yr)))
    elif scope == 'universe':
        # every company-year the DeepSeek vintages label: the classified universe, 2020-2026
        L = pd.read_csv(PROCESSED / 'firm_label_vintages.csv', dtype={'gvkey': str})
        pairs = sorted(set(zip(L.gvkey, L.year.astype(int))))
    else:
        E = load_exposure_table('deepseek')
        D = E[E.theme_broad & (E.labelled >= 50)][['series_id', 'q']]
        H = load_holdings().merge(D, on=['series_id', 'q'])
        pairs = sorted(set(zip(H.gvkey, H.yr)))
    out = []
    for g, y in sorted(pairs, key=lambda t: t[1]):
        if g not in U.index:
            continue
        set_vintage(y)
        rec = U.loc[g].to_dict()
        rec['gvkey'] = g
        out.append({'key': '%s|%d' % (g, y), 'gvkey': g, 'yr': y,
                    'content': C01.PROMPT_A + C01.describe(rec)})
    return out


# ---------------------------------------------------------------- API
KEY_FILE = Path(os.environ.get('OPENAI_API_KEY_FILE', str(ROOT.parent / 'API Key.txt')))


def api_key():
    """The OpenAI key: OPENAI_API_KEY, else the key file. Never printed or logged.

    The file may hold several providers' keys. Only a line that names OpenAI, or a token
    in OpenAI's project-key format, is used; anything ambiguous stops the run rather than
    risk sending another provider's key to OpenAI."""
    k = os.environ.get('OPENAI_API_KEY', '').strip()
    if k:
        return k
    if not KEY_FILE.exists():
        sys.exit('No OPENAI_API_KEY and no key file at %s' % KEY_FILE)
    lines = [ln.strip() for ln in KEY_FILE.read_text(encoding='utf-8', errors='replace').splitlines() if ln.strip()]
    tokens = []
    for ln in lines:
        if 'openai' in ln.lower():
            tokens += [t for t in ln.replace('=', ' ').replace(':', ' ').split() if t.startswith('sk-')]
    if not tokens:
        tokens = [t for ln in lines for t in ln.split() if t.startswith('sk-proj-')]
    tokens = sorted(set(tokens))
    if len(tokens) != 1:
        sys.exit('Could not identify exactly one OpenAI key in %s (found %d candidates). Label the '
                 'line, e.g. "OpenAI: sk-...", or set OPENAI_API_KEY.' % (KEY_FILE, len(tokens)))
    return tokens[0]


class CallLog:
    """Every request and response, as the thesis AI-usage rules require: model, parameters,
    full prompt, raw output, request id, time and tokens. The key is never written."""

    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()

    def write(self, rec):
        line = (json.dumps(rec, ensure_ascii=False) + '\n').encode('utf-8')
        with self.lock, gzip.open(self.path, 'ab') as fh:
            fh.write(line)


def list_models(flt):
    r = requests.get(API + '/models', headers={'Authorization': 'Bearer ' + api_key()}, timeout=60)
    if r.status_code != 200:
        sys.exit('HTTP %d from /models: %s' % (r.status_code, r.text[:200]))
    ids = sorted(m['id'] for m in r.json().get('data', []))
    hits = [i for i in ids if flt.lower() in i.lower()] if flt else ids
    print('\n'.join(hits) if hits else 'no model id contains %r (%d models visible)' % (flt, len(ids)))


class Client:
    def __init__(self, model, max_out, log):
        self.model, self.max_out, self.key, self.log = model, max_out, api_key(), log
        self.drop = set()                 # parameters the model rejected
        self.lock = threading.Lock()
        self.usage = {'in': 0, 'out': 0, 'calls': 0, 'models': set()}

    def ask(self, content, ctx, tries=4):
        err = 'failed'
        for a in range(tries):
            body = {'model': self.model, 'messages': [{'role': 'user', 'content': content}],
                    'response_format': {'type': 'json_object'}, 'temperature': 0,
                    'max_completion_tokens': self.max_out}
            for p in self.drop:
                body.pop(p, None)
            t0 = time.time()
            entry = {'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'task': ctx['task'],
                     'item': ctx['key'], 'attempt': a + 1, 'endpoint': API + '/chat/completions',
                     'request': body}
            try:
                r = requests.post(API + '/chat/completions', json=body, timeout=180,
                                  headers={'Authorization': 'Bearer ' + self.key})
            except requests.RequestException as e:
                err = repr(e)[:150]
                self.log.write({**entry, 'status': None, 'error': err, 'seconds': round(time.time() - t0, 2)})
                time.sleep(3 * (a + 1))
                continue
            try:
                parsed = r.json()
            except ValueError:
                parsed = None
            self.log.write({**entry, 'status': r.status_code, 'request_id': r.headers.get('x-request-id'),
                            'seconds': round(time.time() - t0, 2),
                            'response': parsed if parsed is not None else r.text[:4000]})
            if r.status_code == 400:
                msg = r.text.lower()
                bad = [p for p in ('temperature', 'response_format', 'max_completion_tokens') if p in msg]
                if bad and a < tries - 1:
                    with self.lock:
                        self.drop.update(bad)
                    continue
                return None, 'HTTP 400: ' + r.text[:200]
            if r.status_code in (429, 500, 502, 503) and a < tries - 1:
                time.sleep(5 * (a + 1))
                continue
            if r.status_code != 200 or parsed is None:
                return None, 'HTTP %d: %s' % (r.status_code, r.text[:200])
            j = parsed
            with self.lock:
                self.usage['in'] += j.get('usage', {}).get('prompt_tokens', 0)
                self.usage['out'] += j.get('usage', {}).get('completion_tokens', 0)
                self.usage['calls'] += 1
                if j.get('model'):
                    self.usage['models'].add(j['model'])
            text = (j['choices'][0]['message'].get('content') or '').strip()
            if not text:
                err = 'empty content'
                continue
            try:
                return json.loads(text), None
            except json.JSONDecodeError:
                try:
                    return json.JSONDecoder().raw_decode(text[text.index('{'):])[0], None
                except (ValueError, json.JSONDecodeError):
                    err = 'unparseable: ' + text[:120]
                    continue
        return None, err


def run(model, scope, workers, max_usd_in, max_usd_out, budget):
    out = INTERIM / 'cross_model' / model.replace('/', '_')
    out.mkdir(parents=True, exist_ok=True)
    if scope == 'preevent':
        tasks = [('company_preevent', preevent_records())]
    else:
        tasks = [('fund', fund_records(scope)), ('company', company_records(scope))]
    client = Client(model, 4000, CallLog(out / 'calls.jsonl.gz'))
    started = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    for name, recs in tasks:
        dest = out / ('%s.jsonl' % name)
        done = set()
        if dest.exists():
            for line in open(dest, encoding='utf-8'):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # an answer missing a company role is asked again; the newer line wins
                if not r.get('error') and (name == 'fund' or all(r.get(f) is not None for f in ROLES)):
                    done.add(r['key'])
        todo = [r for r in recs if r['key'] not in done]
        print('%s: %d items, %d cached, %d to call' % (name, len(recs), len(done), len(todo)), flush=True)

        def work(rec):
            spent = client.usage['in'] / 1e6 * max_usd_in + client.usage['out'] / 1e6 * max_usd_out
            if budget and spent >= budget:
                return {'key': rec['key'], 'error': 'budget reached'}
            p, err = client.ask(rec['content'], {'task': name, 'key': rec['key']})
            res = {k: v for k, v in rec.items() if k != 'content'}
            res.update({'model': model, 'error': err})
            if p:
                if name == 'fund':
                    for f in S02.FLAGS:
                        res[f] = S02.yn(p.get(f))
                else:
                    for f in ROLES:
                        res[f] = C01.lev(p.get(f))
                    try:
                        res['ai_sales_share'] = float(p.get('ai_sales_share'))
                    except (TypeError, ValueError):
                        res['ai_sales_share'] = None
            return res
        with open(dest, 'a', encoding='utf-8') as fh, ThreadPoolExecutor(max_workers=workers) as ex:
            for i, res in enumerate(ex.map(work, todo), 1):
                fh.write(json.dumps(res, ensure_ascii=False) + '\n')
                if i % 50 == 0:
                    fh.flush()
                    print('  %d/%d  tokens in=%d out=%d' % (i, len(todo), client.usage['in'], client.usage['out']), flush=True)
    u = client.usage
    cost = u['in'] / 1e6 * max_usd_in + u['out'] / 1e6 * max_usd_out
    print('calls %d, tokens in %d, out %d; at $%.2f/$%.2f per million: about $%.2f'
          % (u['calls'], u['in'], u['out'], max_usd_in, max_usd_out, cost))
    session = {'started_utc': started, 'ended_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               'model_requested': model, 'models_reported_by_api': sorted(u['models']), 'scope': scope,
               'calls': u['calls'], 'input_tokens': u['in'], 'output_tokens': u['out'],
               'price_assumed_usd_per_million': [max_usd_in, max_usd_out], 'cost_estimate_usd': round(cost, 4),
               'parameters_dropped_because_rejected': sorted(client.drop)}
    with open(out / 'sessions.jsonl', 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(session) + '\n')
    write_usage_record(model)


def write_usage_record(model):
    """A readable AI-usage record, computed from the session and call logs."""
    import hashlib
    out = INTERIM / 'cross_model' / model.replace('/', '_')
    sessions = []
    if (out / 'sessions.jsonl').exists():
        sessions = [json.loads(x) for x in open(out / 'sessions.jsonl', encoding='utf-8') if x.strip()]
    entries = 0
    if (out / 'calls.jsonl.gz').exists():
        with gzip.open(out / 'calls.jsonl.gz', 'rt', encoding='utf-8') as fh:
            entries = sum(1 for _ in fh)
    fund_hash = hashlib.sha256(S02.PROMPTS['1'].encode('utf-8')).hexdigest()[:16]
    firm_hash = hashlib.sha256(C01.PROMPT_A.encode('utf-8')).hexdigest()[:16]
    reported = sorted({m for s in sessions for m in s.get('models_reported_by_api', [])})
    year = sessions[0]['started_utc'][:4] if sessions else 'n.d.'
    rows = [
        ('Tool', 'OpenAI API, model requested `%s`; model reported by the API: %s'
         % (model, ', '.join('`%s`' % m for m in reported) or 'none')),
        ('Dates (UTC)', '; '.join('%s to %s' % (s['started_utc'], s['ended_utc']) for s in sessions)),
        ('Purpose', 'Label AI-role statements in fund prospectuses and the AI roles of portfolio '
                    'companies with a second model family, using the classifier prompts unchanged; '
                    'first to measure how far the labels depend on the model, then as the thesis measurement'),
        ('Inputs', 'Fund prompt (src/fund_claims/s02_classify.py, first wording, SHA-256 %s...) and '
                   'company prompt (src/firm_labels/c01_classify.py PROMPT_A with describe(), SHA-256 %s...); '
                   'the same filing text, patent profiles and as-of financials the original classifier received, including the pre-event information sets of the event analysis'
         % (fund_hash, firm_hash)),
        ('Parameters', 'JSON response format; at most 4000 output tokens; temperature 0 requested%s'
         % ('; the model rejected: %s, so those were dropped and the model default applies (outputs can '
            'vary between runs)' % ', '.join(sorted({p for s in sessions for p in s.get('parameters_dropped_because_rejected', [])}))
            if any(s.get('parameters_dropped_because_rejected') for s in sessions) else '')),
        ('Volume', '%d successful calls; %d input and %d output tokens; %d logged request/response entries'
         % (sum(s['calls'] for s in sessions), sum(s['input_tokens'] for s in sessions),
            sum(s['output_tokens'] for s in sessions), entries)),
        ('Cost', 'about USD %.2f at the assumed list price (USD per million tokens: %s); the OpenAI '
                 'billing page is authoritative' % (sum(s['cost_estimate_usd'] for s in sessions),
                                                    ', '.join(str(s['price_assumed_usd_per_million']) for s in sessions[:1]))),
        ('Records', 'calls.jsonl.gz: every request (model, parameters, full prompt) and the raw response, '
                    'with request id, time and token usage; fund.jsonl, company.jsonl and company_preevent.jsonl: parsed labels; '
                    'sessions.jsonl: one line per run. The API key is not stored'),
        ('Use in the thesis', 'These labels are the thesis measurement (chosen on the '
                              'public capability scores of the two models, after the 280-item agreement sample and before any '
                              'downstream estimate on them); the DeepSeek labels are reported alongside as the '
                              'secondary measurement (src/analysis/robustness/luna_measures.py, '
                              'cross_model_downstream.py; agreement in results/tables/robustness/'
                              'cross_model_agreement.csv). Agreement between two models is a consistency check, not accuracy'),
        ('Verification', 'Responses are parsed by code and stored unedited; the comparison is computed '
                         'from the stored responses; accuracy is judged against human coding of the '
                         'stratified sample'),
    ]
    name = DISPLAY.get(model, model)
    text = ['# AI usage record: %s labels' % name, '', '| Field | Entry |', '|---|---|']
    text += ['| %s | %s |' % r for r in rows]
    text += ['', 'APA 7 reference (check the version name OpenAI publishes for this model):', '',
             'OpenAI. (%s). *%s* [Large language model; API model %s]. https://platform.openai.com/docs/models' % (year, name, model), '']
    (out / 'AI_USAGE_RECORD.md').write_text('\n'.join(text), encoding='utf-8')
    print('usage record written to', out / 'AI_USAGE_RECORD.md')


def estimate():
    for scope in ('sample', 'all'):
        for name, recs in (('fund', fund_records(scope)), ('company', company_records(scope))):
            chars = sum(len(r['content']) for r in recs)
            print('%-7s %-8s %6d calls, about %5.1f million input tokens (chars/4), '
                  'about %4.2f million output tokens'
                  % (scope, name, len(recs), chars / 4e6, len(recs) * 250 / 1e6))


# ---------------------------------------------------------------- comparison
def kappa(a, b, w=None):
    """Cohen's kappa; with design weights `w`, agreement and margins are weighted."""
    a, b = pd.Series(a).reset_index(drop=True), pd.Series(b).reset_index(drop=True)
    w = pd.Series(1.0, index=a.index) if w is None else pd.Series(w).reset_index(drop=True).astype(float)
    ok = a.notna() & b.notna()
    a, b, w = a[ok], b[ok], w[ok]
    if not len(a):
        return np.nan, 0
    W = w.sum()
    cats = sorted(set(a) | set(b))
    po = float(w[a == b].sum() / W)
    pe = sum(float(w[a == c].sum() / W) * float(w[b == c].sum() / W) for c in cats)
    return ((po - pe) / (1 - pe) if pe < 1 else np.nan), int(len(a))


def wmean(x, w):
    x, w = pd.Series(x).reset_index(drop=True).astype(float), pd.Series(w).reset_index(drop=True).astype(float)
    ok = x.notna()
    return float((x[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else np.nan


def read_jsonl(p):
    rows = []
    if p.exists():
        for line in open(p, encoding='utf-8'):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not r.get('error'):
                rows.append(r)
    D = pd.DataFrame(rows)
    return D.drop_duplicates('key', keep='last') if 'key' in D.columns else D


def compare(model):
    base = INTERIM / 'cross_model' / model.replace('/', '_')
    rows = []
    # funds: second model vs the classifier's first prompt (same prompt), and the classifier's
    # own two prompts as the within-model benchmark
    L = read_jsonl(base / 'fund.jsonl')
    if len(L):
        d1 = read_jsonl(INTERIM / 'fund_claims' / 'labels_run1.jsonl').drop_duplicates(['accession', 'series_id', 'evidence_sha256'], keep='last')
        d2 = read_jsonl(INTERIM / 'fund_claims' / 'labels_run2.jsonl').drop_duplicates(['accession', 'series_id', 'evidence_sha256'], keep='last')
        k = ['accession', 'series_id', 'evidence_sha256']
        M = L.merge(d1, on=k, suffixes=('', '_ds1')).merge(d2, on=k, suffixes=('', '_ds2'))
        KF = pd.read_csv(VAL / 'KEY_fund_claims_model_labels_DO_NOT_OPEN_BEFORE_CODING.csv', dtype=str)
        KF['w'] = KF.weight.astype(float)
        M = M.merge(KF[['accession', 'series_id', 'w']], on=['accession', 'series_id'], how='left')
        S = M[M.w.notna()]
        for f in FUND_FLAGS:
            for lab, a, b in (('second model vs classifier (same prompt)', f, f + '_ds1'),
                              ('classifier: prompt 1 vs prompt 2', f + '_ds1', f + '_ds2')):
                kp, n = kappa(M[a], M[b])
                kw, nw = kappa(S[a], S[b], S.w)
                rows.append({'side': 'fund', 'item': f, 'comparison': lab, 'n': n,
                             'agreement': float((M[a] == M[b]).mean()), 'kappa': kp,
                             'n_sample': nw, 'agreement_weighted': wmean(S[a] == S[b], S.w), 'kappa_weighted': kw,
                             'share_yes_first_weighted': wmean(S[a] == 'Y', S.w),
                             'share_yes_second_weighted': wmean(S[b] == 'Y', S.w)})
    C = read_jsonl(base / 'company.jsonl')
    if len(C):
        parts = []
        for y in sorted(C.yr.unique()):
            for run in ('3', '4'):
                p = VINT / '01_classify' / str(int(y)) / ('labels_ev_run%s.jsonl' % run)
                R = read_jsonl(p)
                if len(R):
                    R = R.assign(yr=int(y), run=run)[['gvkey', 'yr', 'run'] + ROLES + ['ai_sales_share']]
                    parts.append(R)
        DS = pd.concat(parts, ignore_index=True)
        DS['gvkey'] = DS.gvkey.astype(str)
        D3 = DS[DS.run == '3'].drop(columns='run').drop_duplicates(['gvkey', 'yr'], keep='last')
        D4 = DS[DS.run == '4'].drop(columns='run').drop_duplicates(['gvkey', 'yr'], keep='last')
        C['gvkey'] = C.gvkey.astype(str)
        C['yr'] = C.yr.astype(int)
        M = C.merge(D3, on=['gvkey', 'yr'], suffixes=('', '_ds3')).merge(D4, on=['gvkey', 'yr'], suffixes=('', '_ds4'))
        KC = pd.read_csv(VAL / 'KEY_company_roles_model_labels_DO_NOT_OPEN_BEFORE_CODING.csv', dtype={'gvkey': str})
        KC['w'] = KC.weight.astype(float)
        KC['yr'] = KC.yr.astype(int)
        M = M.merge(KC[['gvkey', 'yr', 'w']], on=['gvkey', 'yr'], how='left')
        S = M[M.w.notna()]
        for f in ROLES:
            for lab, a, b in (('second model vs classifier (same prompt)', f, f + '_ds3'),
                              ('classifier: prompt A vs prompt B', f + '_ds3', f + '_ds4')):
                ya, yb = M[a].ge(2).where(M[a].notna()), M[b].ge(2).where(M[b].notna())
                sa, sb = S[a].ge(2).where(S[a].notna()), S[b].ge(2).where(S[b].notna())
                kp, n = kappa(ya, yb)
                kw, nw = kappa(sa, sb, S.w)
                ks, _ = kappa(M[a], M[b])
                rows.append({'side': 'company', 'item': f + ' >= 2', 'comparison': lab, 'n': n,
                             'agreement': float((ya == yb).mean()), 'kappa': kp, 'kappa_0to3': ks,
                             'n_sample': nw, 'agreement_weighted': wmean(sa == sb, S.w), 'kappa_weighted': kw,
                             'share_yes_first_weighted': wmean(sa, S.w), 'share_yes_second_weighted': wmean(sb, S.w)})
        r = M[['ai_sales_share', 'ai_sales_share_ds3']].astype(float).corr().iloc[0, 1]
        rows.append({'side': 'company', 'item': 'ai_sales_share', 'comparison': 'second model vs classifier (correlation)',
                     'n': int(M[['ai_sales_share', 'ai_sales_share_ds3']].notna().all(axis=1).sum()), 'agreement': np.nan, 'kappa': np.nan,
                     'correlation': float(r)})
    R = pd.DataFrame(rows)
    R.insert(0, 'model', model)
    R.to_csv(TABLES / 'cross_model_agreement.csv', index=False, float_format='%.4f')
    with pd.option_context('display.width', 220):
        print(R.round(3).to_string())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('action', choices=['estimate', 'models', 'run', 'compare'])
    ap.add_argument('--model')
    ap.add_argument('--scope', choices=['sample', 'all', 'luna-theme', 'universe', 'preevent'], default='sample')
    ap.add_argument('--filter', default='luna')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--price-in', type=float, default=0.10, help='USD per million input tokens, for the running estimate')
    ap.add_argument('--price-out', type=float, default=0.50, help='USD per million output tokens')
    ap.add_argument('--budget', type=float, default=1.0, help='stop calling once the estimated spend reaches this many USD')
    a = ap.parse_args(argv)
    if a.action == 'estimate':
        estimate()
    elif a.action == 'models':
        list_models(a.filter)
    elif a.action == 'run':
        if not a.model:
            sys.exit('--model is required')
        run(a.model, a.scope, a.workers, a.price_in, a.price_out, a.budget)
    else:
        if not a.model:
            sys.exit('--model is required')
        compare(a.model)


if __name__ == '__main__':
    main()

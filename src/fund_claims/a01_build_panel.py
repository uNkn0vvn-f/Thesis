"""The full fund x filing panel, including the filings that say nothing about AI.

Everything upstream is conditioned on the AI keyword screen: a filing only reaches the
classifier if its Principal Investment Strategies text mentions AI. That suffices for
measuring what a claim says, but not for anything with a time dimension, because a fund
that deletes its AI language simply stops appearing, and a fund-level `ever` column would
mark it as a claimant for good.

This script therefore builds the panel over EVERY 485BPOS fund-filing with an
extractable strategy section, carrying explicit zeros for the filings that mention no AI,
and provides an as-of view: the claim state of each fund at its last filing on or before
a chosen date. That, not `ever`, is what a dated design needs.
"""
import glob, gzip, io, json, os, re

import numpy as np
import pandas as pd
from src.fund_claims.asof import MAX_AGE_DAYS                                # noqa: E402

from src import paths as tp  # noqa: E402

OUT = tp.out('fund_claims')
LOGS = tp.logs('fund_claims')
os.makedirs(OUT, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)

# s00_attribute gives each fund in a multi-fund filing its own strategy text; the raw
# corpus left 6,887 filings' text with no series id and those funds with none.
FUNDS = tp.out('fund_claims', 'funds_attributed')
HDRS = tp.data('485bpos_headers')
LABELS = tp.out('fund_claims', 'labels_final_filing.parquet')
CACHE = os.path.join(OUT, '_all_filings.parquet')

CLAIMS = ['uses_ai', 'uses_ai_hedged', 'uses_ai_support', 'invests_ai_theme',
          'theme_broad', 'theme_strict', 'theme_incidental', 'chain_developer',
          'chain_enabler', 'chain_adopter', 'chain_benefit_vague',
          'chain_unspecified', 'falsifiable_test', 'strategy_text_risk_only']
SHOCKS = {'chatgpt': '2022-11-30', 'names_rule': '2023-09-20',
          'sec_ai_washing': '2024-03-18'}


def single_series():
    out = {}
    for p in sorted(glob.glob(os.path.join(HDRS, '*.jsonl.gz'))):
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
                        out[r.get('accession')] = ser[0]['series_id']
        except Exception:
            continue
    return out


def scan_all():
    inputs = (sorted(glob.glob(os.path.join(HDRS, '*.jsonl.gz')))
              + sorted(glob.glob(os.path.join(FUNDS, '*.jsonl.gz'))))
    if tp.cache_valid(CACHE, inputs):
        return pd.read_parquet(CACHE)
    hdr = single_series()
    rows = []
    for p in sorted(glob.glob(os.path.join(FUNDS, '*.jsonl.gz'))):
        try:
            with gzip.open(p, 'rt', encoding='utf-8') as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except Exception:
                        break
                    if not (r.get('strategy') or '').strip():
                        continue
                    sid = (r.get('series_id') or '').strip()
                    resolved = bool(sid)
                    if not resolved:
                        sid = hdr.get(r.get('accession'), '')
                        resolved = bool(sid)
                    rows.append((r.get('accession'), sid or None, resolved,
                                 r.get('series_name'), r.get('cik'), r.get('date'),
                                 bool(r.get('ai_strategy')),
                                 len(r.get('strategy') or '')))
        except Exception:
            continue
    D = pd.DataFrame(rows, columns=['accession', 'series_id', 'series_resolved',
                                    'series_name', 'cik', 'date', 'ai_keyword',
                                    'strategy_chars'])
    D.to_parquet(CACHE, index=False)
    tp.cache_written(CACHE, inputs)
    return D


def main():
    buf = io.StringIO()

    def w(s=''):
        buf.write(s + '\n')
        try:
            print(s)
        except UnicodeEncodeError:
            print(s.encode('ascii', 'replace').decode())

    D = scan_all()
    D['date'] = pd.to_datetime(D.date, errors='coerce')
    D = D[D.date.notna()].drop_duplicates(['accession', 'series_id'])

    L = pd.read_parquet(LABELS)
    keep = ['accession', 'series_id'] + [c for c in CLAIMS if c in L.columns]
    P = D.merge(L[keep], on=['accession', 'series_id'], how='left')
    for c in CLAIMS:
        if c in P.columns:
            P[c] = P[c].astype('boolean').fillna(False).astype(bool)
        else:
            P[c] = False
    P['classified'] = P.accession.isin(set(L.accession)) & P.ai_keyword

    w('=' * 96)
    w('FULL FUND x FILING PANEL, WITH EXPLICIT ZEROS')
    w('=' * 96)
    w('  fund-filings with an extractable strategy : %d' % len(P))
    w('  distinct series                           : %d' % P.series_id.nunique())
    w('  span                                      : %s .. %s'
      % (P.date.min().date(), P.date.max().date()))
    w('  of which the AI keyword fires             : %d (%.2f%%)'
      % (int(P.ai_keyword.sum()), 100 * P.ai_keyword.mean()))
    w('  of which carry a classified label         : %d' % int(P.classified.sum()))
    w('  series_id unresolved                      : %d' % int((~P.series_resolved).sum()))
    w()
    w('  Everything else is an explicit zero. In the screened table those rows did not')
    w('  exist, so a fund that stopped writing about AI stayed a claimant forever.')
    w()

    R = P[P.series_resolved & P.series_id.notna()].sort_values(['series_id', 'date'])

    # ---------------- exits ----------------
    w('-' * 96)
    w('EXITS: FUNDS THAT STOPPED CLAIMING')
    w('-' * 96)
    g = R.groupby('series_id')
    last = g.tail(1).set_index('series_id')
    ever_claim = g[['uses_ai', 'theme_broad']].any().any(axis=1)
    last_claim = last.uses_ai | last.theme_broad
    exited = ever_claim & ~last_claim.reindex(ever_claim.index).fillna(False)
    w('  funds that EVER made an affirmative claim : %d' % int(ever_claim.sum()))
    w('  of those, their LATEST filing makes none  : %d' % int(exited.sum()))
    ever_kw = g.ai_keyword.any()
    last_kw = last.ai_keyword.reindex(ever_kw.index).fillna(False)
    w('  funds whose AI keyword fired once but not in their latest filing : %d'
      % int((ever_kw & ~last_kw).sum()))
    w()

    # ---------------- as-of ----------------
    w('-' * 96)
    w('AS-OF TREATMENT vs EVER, AT THREE DATES')
    w('-' * 96)
    w('  %-14s %-12s %10s %10s %10s %10s'
      % ('cutoff', '', 'uses_ai', 'theme_broad', 'theme_str', 'funds'))
    asof_rows = []
    for nm, d in SHOCKS.items():
        cut = pd.Timestamp(d)
        pre = R[R.date <= cut]
        snap = pre.sort_values('date').groupby('series_id').tail(1)
        # a fund whose last prospectus is over 16 months old at the cutoff is not a live
        # fund with that claim - it stopped filing, or its update was missed (asof.py)
        snap = snap[(cut - snap.date).dt.days <= MAX_AGE_DAYS]
        asof_rows.append(snap.assign(cutoff=nm, cutoff_date=cut))
        w('  %-14s %-12s %10d %10d %10d %10d'
          % (nm, str(cut.date()), int(snap.uses_ai.sum()),
             int(snap.theme_broad.sum()), int(snap.theme_strict.sum()), len(snap)))
    ev = g[['uses_ai', 'theme_broad', 'theme_strict']].any()
    w('  %-14s %-12s %10d %10d %10d %10d'
      % ('EVER', '2020-2026', int(ev.uses_ai.sum()), int(ev.theme_broad.sum()),
         int(ev.theme_strict.sum()), len(ev)))
    w()
    w('  The gap between the ChatGPT row and the EVER row is the look-ahead that an')
    w('  `ever` fund-level flag would carry into every design that has a date in it.')
    w()

    A = pd.concat(asof_rows, ignore_index=True)
    P.to_parquet(os.path.join(OUT, 'panel_fund_filing.parquet'), index=False)
    A.to_parquet(os.path.join(OUT, 'asof_treatment.parquet'), index=False)
    ex = pd.DataFrame({'ever_claim': ever_claim, 'exited': exited})
    ex.to_parquet(os.path.join(OUT, 'exits.parquet'))
    w('wrote out/panel_fund_filing.parquet (%d rows - the comparison group is IN here)'
      % len(P))
    w('wrote out/asof_treatment.parquet    (%d rows, %d cutoffs)'
      % (len(A), len(SHOCKS)))
    w('wrote out/exits.parquet             (%d series)' % len(ex))

    with io.open(os.path.join(LOGS, 'a01_build_panel.txt'), 'w',
                 encoding='utf-8') as fh:
        fh.write(buf.getvalue())


if __name__ == '__main__':
    main()

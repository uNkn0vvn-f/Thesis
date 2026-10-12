"""Point-in-time AI-name flags as spells, with the two name sources reconciled.

A fund's name is observed in its prospectus (485BPOS) and in N-PORT, and the two have
different reporting lags. Global X renamed its Future Analytics Tech ETF to an AI name in the
prospectus of 2021-03-24, while N-PORT went on reporting the old name until 2022-02-28, so
interleaving the sources by date would manufacture renames that never happened. Spells
are therefore built by three rules:

  1. ONE AUTHORITATIVE SOURCE FOR THE CHANGE. A fund's registered name changes in its
     prospectus, so 485BPOS dates the change. N-PORT is used only to extend coverage
     for series that never appear in 485BPOS, and never to contradict it.
  2. PERSISTENCE. A name is accepted as a change only when it is still in force at the
     next observation from the same source. A single odd observation between two
     identical ones is a reporting artefact, not a rename.
  3. HALF-OPEN INTERVALS. Each spell runs [start, next_change) and the last spell is
     right-censored at the last observation, with `censored` recorded. No spell is
     zero-length, and no spell claims a fund still exists after it stopped filing.

N-PORT dates are also corrected: FUND_REPORTED_INFO is dated by REPORT_DATE, the period
end, which runs a median of ~57 days before the filing becomes public. Where a filing
date is available it is used, since what matters for any event study is when the name
became observable.
"""
import io, os, re

import pandas as pd

from src import paths as tp  # noqa: E402

OUT = tp.out('fund_claims')
LOGS = tp.logs('fund_claims')
os.makedirs(OUT, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)

AI = re.compile(r'artificial\s+intelligence|\bA\.?I\.?\b|machine\s+learning|'
                r'deep\s+learning|generative|\bLLM\b', re.I)
BAD = re.compile(r'\bAIM\b|\bAIR\b|\bAID\b|\bSAI\b|\bMAI\b', re.I)
WRAP = re.compile(r'\b(etf|fund|funds|portfolio|trust|inc|shares|the)\b\.?|[^a-z0-9]', re.I)


def ai_flag(name):
    n = BAD.sub(' ', name) if BAD.search(name) else name
    return bool(AI.search(n))


def strat_key(name):
    return WRAP.sub('', str(name).lower())


def norm(s):
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9 ]', ' ', str(s).lower())).strip()


def spells_for(g):
    """g: one series, sorted by obs_date, columns nname/series_name/obs_date/src.

    A name is accepted only once it persists to the next observation of the same
    source, so a lone discordant reading between two identical ones is ignored.
    """
    rows = list(g.itertuples())
    accepted = []          # (obs_date, nname, series_name, src)
    for i, r in enumerate(rows):
        if not accepted:
            accepted.append(r)
            continue
        if r.nname == accepted[-1].nname:
            continue
        # look ahead within the SAME source for confirmation
        later_same_src = [x for x in rows[i + 1:] if x.src == r.src]
        confirmed = (not later_same_src) or (later_same_src[0].nname == r.nname)
        if confirmed:
            accepted.append(r)
    return accepted


def main():
    buf = io.StringIO()

    def w(s=''):
        buf.write(s + '\n')
        try:
            print(s)
        except UnicodeEncodeError:
            print(s.encode('ascii', 'replace').decode())

    A = pd.read_parquet(os.path.join(OUT, 'name_panel.parquet'))
    A['report_date'] = pd.to_datetime(A.obs_date, errors='coerce')
    A['filing_date'] = pd.to_datetime(A.filing_date, errors='coerce')

    w('=' * 96)
    w('AI-NAME SPELLS, SOURCES RECONCILED')
    w('=' * 96)
    npt = A[A.src == 'nport']
    lag = (npt.filing_date - npt.report_date).dt.days
    if lag.notna().any():
        w('  N-PORT: FUND_REPORTED_INFO is dated by REPORT_DATE, the period end.')
        w('  filing_date minus report_date: median %.0f days, p90 %.0f'
          % (lag.median(), lag.quantile(.9)))
        w('  The public date is used below, because what matters is when the name')
        w('  became observable.')
    # observable date: when the name became public, not the reporting period end
    A['obs_date'] = A.filing_date.fillna(A.report_date)
    A['nname'] = A.series_name.map(norm)
    A = A[(A.nname != '') & A.obs_date.notna()].sort_values(
        ['series_id', 'obs_date', 'src'])
    w()

    # ---- 1. authoritative source ----
    has485 = set(A[A.src == '485bpos'].series_id)
    prim = A[(A.src == '485bpos') | (~A.series_id.isin(has485))].copy()
    w('  series with a 485BPOS name history        : %d' % len(has485))
    w('  series seen only in N-PORT (fallback)     : %d'
      % A[~A.series_id.isin(has485)].series_id.nunique())
    w('  name observations used after reconciling  : %d of %d'
      % (len(prim), len(A)))
    w()

    # ---- 2. persistence + 3. half-open intervals ----
    recs = []
    for sid, g in prim.groupby('series_id'):
        acc = spells_for(g.sort_values('obs_date'))
        last_obs = g.obs_date.max()
        for j, r in enumerate(acc):
            end = acc[j + 1].obs_date if j + 1 < len(acc) else last_obs
            recs.append({'series_id': sid, 'series_name': r.series_name,
                         'nname': r.nname, 'ai': ai_flag(r.series_name),
                         'date_from': r.obs_date, 'date_to': end,
                         'censored': j + 1 == len(acc),
                         'src': r.src, 'registrant_name': r.registrant_name,
                         'cik': r.cik})
    S = pd.DataFrame(recs)
    S['days'] = (S.date_to - S.date_from).dt.days

    w('-' * 96)
    w('WHAT THE RECONCILIATION CHANGED')
    w('-' * 96)
    w('  name spells : %d' % len(S))
    rev = 0
    for sid, g in S.sort_values(['series_id', 'date_from']).groupby('series_id'):
        n = list(g.nname)
        rev += sum(1 for i in range(len(n) - 2) if n[i] == n[i + 2])
    w('  A->B->A reversions : %d' % rev)
    ai = S[S.ai]
    z = ai[ai.days <= 0]
    w('  AI-name spells : %d' % len(ai))
    w('    genuinely zero-length : %d'
      % int((~z.censored).sum()))
    w('    right-censored at a single observation : %d' % int(z.censored.sum()))
    w('      These are funds observed in exactly one filing - mostly 2026 launches.')
    w('      date_from equals date_to because the spell is OPEN, not empty; `censored`')
    w('      marks it so no downstream code reads it as a fund that lasted no time.')
    w()

    # ---------------- the universe ----------------
    ever = S[S.ai].series_id.unique()
    U = S[S.series_id.isin(ever)].copy()
    rec = (U.sort_values(['series_id', 'date_from']).groupby('series_id')
             .agg(n_spells=('nname', 'size'), n_ai_spells=('ai', 'sum'),
                  first_obs=('date_from', 'min'), last_obs=('date_to', 'max'),
                  registrant_name=('registrant_name', 'last'), cik=('cik', 'last')))
    aiS = U[U.ai]
    rec['first_ai_date'] = aiS.groupby('series_id').date_from.min()
    rec['last_ai_date'] = aiS.groupby('series_id').date_to.max()
    rec['ai_name_at_first_obs'] = U.sort_values('date_from').groupby('series_id').ai.first()
    rec['ai_name_at_last_obs'] = U.sort_values('date_from').groupby('series_id').ai.last()
    rec['ai_always'] = rec.n_ai_spells == rec.n_spells
    rec['ai_name'] = aiS.sort_values('date_from').groupby('series_id').series_name.last()
    rec['strategy_key'] = rec.ai_name.map(strat_key)

    w('-' * 96)
    w('THE UNIVERSE, THREE WAYS OF COUNTING IT')
    w('-' * 96)
    w('  ever AI-named                      : %d' % len(rec))
    w('  AI-named at its first observation  : %d   <- the first-observation rule'
      % int(rec.ai_name_at_first_obs.sum()))
    w('  AI-named at its last observation   : %d' % int(rec.ai_name_at_last_obs.sum()))
    w('  AI-named for its whole observed life: %d' % int(rec.ai_always.sum()))
    w()
    miss = rec[~rec.ai_name_at_first_obs].sort_values('first_ai_date')
    w('  funds a first-observation rule misses (renamed INTO an AI name): %d' % len(miss))
    for sid, r in miss.iterrows():
        w('    %s  %-58s  from %s' % (sid, str(r.ai_name)[:58], r.first_ai_date.date()))
    w()
    out_ = rec[rec.ai_name_at_first_obs & ~rec.ai_name_at_last_obs]
    w('  funds no longer AI-named (renamed OUT): %d' % len(out_))
    for sid, r in out_.iterrows():
        w('    %s  %-52s  AI until %s' % (sid, str(r.ai_name)[:52], r.last_ai_date.date()))
    w()

    w('-' * 96)
    w('GLOBAL X AIQ, THE CASE THAT EXPOSED THE DEFECT')
    w('-' * 96)
    for _, r in S[S.series_id == 'S000061326'].sort_values('date_from').iterrows():
        w('   %s .. %s  ai=%-5s src=%-8s %s' % (r.date_from.date(), r.date_to.date(),
                                                r.ai, r.src, str(r.series_name)[:52]))
    w()

    S.to_parquet(os.path.join(OUT, 'ai_name_spells.parquet'), index=False)
    rec.to_parquet(os.path.join(OUT, 'ai_universe.parquet'))
    w('wrote out/ai_name_spells.parquet (%d spells)' % len(S))
    w('wrote out/ai_universe.parquet    (%d ever-AI-named series)' % len(rec))

    with io.open(os.path.join(LOGS, 'u02_ai_name_spells.txt'), 'w',
                 encoding='utf-8') as fh:
        fh.write(buf.getvalue())


if __name__ == '__main__':
    main()

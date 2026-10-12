"""Fund-quarter AI exposure on the direct firm label, benchmark-adjusted.

This is the verification layer of the pipeline. The firm-level AI measure is the direct label - a 0-3
intensity on each value-chain position plus an estimated share of revenue from AI - rather
than the union of three proxies (G06N patents, an enabler SIC list, a resume-based
employee share). The proxies tagged 61.9% of held value as AI, so a fund making no AI
claim already looked 34% AI and no comparison could separate anything.

FOUR MEASURES PER FUND-QUARTER, all value-weighted over the part of the portfolio where
the label is observed, with the covered share carried alongside:

    ai_rev_w      mean AI REVENUE SHARE of the holdings. One unit for every firm and
                  every position, and the unit 217 prospectus filings choose for
                  themselves when they set an inclusion test as a revenue threshold.
    share_core    share of value in firms that ARE AI companies (developer or enabler
                  >= 2, or over half their revenue from AI)
    share_<pos>   share of value in firms scoring >= 2 on each of the four positions.
                  This is what makes the diagonal testable.

BENCHMARK. Each measure is compared with a size-decile x sector expected value, so a fund
that simply owns the largest technology names scores zero abnormal exposure. The benchmark
is US-only because Compustat Global has no shares outstanding; the raw share is global and
travels alongside every abnormal figure.

COVERAGE TRAVELS WITH EVERYTHING. A fund of funds, a swap vehicle or a bond fund has
nothing to look through to, and a zero from one of those is a measurement boundary rather
than a finding.
"""
import io, os

import numpy as np
import pandas as pd
from src import paths as tp  # noqa: E402

from src.fund_claims.asof import expire, MAX_AGE_DAYS                        # noqa: E402

# EXP_OUT / LAB may be overridden so the exposure can be built on a different firm
# label - for example, a pre-shock label - without touching these outputs.
OUT = os.environ.get('EXP_OUT') or tp.out('exposure')
# When EXP_OUT is set the log goes INSIDE it, not beside it. Writing to `OUT/../logs`
# gave every tag the same path, so the chatgpt and deepseek runs silently overwrote each
# other's log and only the last one survived.
LOGS = os.path.join(OUT, 'logs') if os.environ.get('EXP_OUT') \
    else tp.logs('exposure')
os.makedirs(OUT, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)

HOLD = tp.out('holdings', 'holdings_bridged.parquet')
VINT = tp.out('firm_labels', 'vintages', '02_label')
# The exposure is POINT-IN-TIME: a fund's holdings in 2022 are weighted by the firm label
# built from filings public at the end of 2021. LAB=<path> instead applies that one label
# to every quarter, which is how a pre-event label is held fixed across its event window.
LAB = os.environ.get('LAB')
SINGLE = LAB is not None
DEST = 'fund_quarter_exposure.parquet'


UNI = tp.out('firm_labels', 'firm_universe.parquet')
SQ = tp.out('holdings', 'stock_quarter.parquet')
PANEL = tp.out('fund_claims', 'panel_fund_filing.parquet')

POS = ('developer', 'enabler', 'adopter')
# `tech` is here to be CONTROLLED FOR, not to be claimed. A fund holding AI holds
# technology - the two portfolio shares correlate 0.675 - so a reader of the ChatGPT
# result alone is entitled to ask whether the flows followed AI or followed the sector.
# The two-shock design answers that jointly, through the sign flip, but only jointly.
# Carrying a technology share on the panel lets every design answer it directly and
# within its own window. It is built from the universe's SIC-derived `sector` and uses
# no LLM label, so it cannot inherit an error from the thing it is controlling.
MEAS = ['ai_rev', 'core'] + list(POS) + ['tech']
CLAIMS = ['uses_ai', 'theme_broad', 'theme_strict', 'theme_incidental',
          'chain_developer', 'chain_enabler', 'chain_adopter', 'chain_benefit_vague',
          'chain_unspecified', 'falsifiable_test']


def load_labels(cols):
    """One label per calendar year, or the single label LAB.

    Returned long with `yr`, so the merge on to holdings is on (gvkey, year) and a firm
    enters or leaves the AI definition as its own filings say it did."""
    if SINGLE:
        L = pd.read_parquet(LAB)
        L['gvkey'] = L.gvkey.astype(str)
        return L[cols], None
    rows = []
    for y in sorted(int(d) for d in os.listdir(VINT) if d.isdigit()):
        p = os.path.join(VINT, str(y), 'firm_ai_labels.parquet')
        if not os.path.exists(p):
            continue
        x = pd.read_parquet(p, columns=cols)
        x['gvkey'] = x.gvkey.astype(str)
        x['yr'] = y
        rows.append(x)
    assert rows, 'no vintages in %s - build them first (python run.py prepare)' % VINT
    L = pd.concat(rows, ignore_index=True)
    assert not L.duplicated(['gvkey', 'yr']).any(), 'a gvkey appears twice in one vintage'
    return L, sorted(L.yr.unique())


def main():
    buf = io.StringIO()

    def w(s=''):
        buf.write(s + '\n')
        try:
            print(s)
        except UnicodeEncodeError:
            print(s.encode('ascii', 'replace').decode())

    L, years = load_labels(['gvkey', 'ai_rev', 'ai_core'] + list(POS))
    # `filing_date` is carried through because `report_date` alone cannot say when a
    # portfolio became PUBLIC: N-PORT allows roughly 60 days, so a portfolio dated
    # 2022-11-30 is not visible until late January. Selecting a pre-shock treatment on
    # report_date alone picked a filing submitted AFTER the shock for 49% of funds.
    H = pd.read_parquet(HOLD, columns=['series_id', 'q', 'gvkey', 'value',
                                       'report_date', 'filing_date', 'net_assets',
                                       'series_name'])
    H['filing_date'] = pd.to_datetime(H.filing_date)
    H = H[H.value.notna() & (H.value > 0) & H.gvkey.notna()].copy()
    H['gvkey'] = H.gvkey.astype(str)
    if SINGLE:
        H = H.merge(L, on='gvkey', how='left')
    else:
        H['yr'] = H.q.str.slice(0, 4).astype(int)
        miss = sorted(set(H.yr) - set(years))
        assert not miss, ('no label vintage for %s; build it before the exposure panel '
                          'or those quarters silently lose every AI firm' % miss)
        H = H.merge(L, on=['gvkey', 'yr'], how='left')
    H['seen'] = H.ai_core.notna()
    H['m_core'] = H.ai_core.astype('boolean').fillna(False).astype(float) * 100
    for p in POS:
        H['m_' + p] = (H[p].fillna(-1) >= 2).astype(float) * 100
    H['m_ai_rev'] = H.ai_rev
    SEC = pd.read_parquet(UNI, columns=['gvkey', 'sector'])
    SEC['gvkey'] = SEC.gvkey.astype(str)
    H = H.merge(SEC, on='gvkey', how='left')
    H['m_tech'] = (H.sector == 'tech').astype(float) * 100

    w('=' * 96)
    w('FUND-QUARTER AI EXPOSURE, ON THE DIRECT FIRM LABEL'
      + ('  [SINGLE LABEL]' if SINGLE else '  [ANNUAL VINTAGES, POINT-IN-TIME]'))
    w('=' * 96)
    w('  holding rows %d, firms %d, labelled %.1f%% of rows and %.1f%% of value'
      % (len(H), H.gvkey.nunique(), 100 * H.seen.mean(),
         100 * H.loc[H.seen, 'value'].sum() / H.value.sum()))
    w()

    # ---------------- firm-level benchmark buckets ----------------
    S = pd.read_parquet(SQ, columns=['gvkey', 'q', 'bucket'])
    S['gvkey'] = S.gvkey.astype(str)
    H = H.merge(S, on=['gvkey', 'q'], how='left')
    w('  rows with a benchmark bucket : %.1f%% of rows, %.1f%% of value'
      % (100 * H.bucket.notna().mean(),
         100 * H.loc[H.bucket.notna(), 'value'].sum() / H.value.sum()))

    # bucket means are built on the label in use, not inherited
    F = H[H.seen & H.bucket.notna()].drop_duplicates(['gvkey', 'q'])
    BM = (F.groupby(['q', 'bucket'])[['m_' + m for m in MEAS]].mean().reset_index()
          .rename(columns={'m_' + m: 'e_' + m for m in MEAS}))
    w('  benchmark cells built on the label : %d (q x bucket)' % len(BM))
    w()
    H = H.merge(BM, on=['q', 'bucket'], how='left')

    # ---------------- aggregate to the fund-quarter ----------------
    g = H.groupby(['series_id', 'q'], sort=False)
    base = g.agg(n_holdings=('gvkey', 'size'), port_value=('value', 'sum'),
                 report_date=('report_date', 'first'),
                 # the EARLIEST filing for the cell: when the portfolio first became
                 # public. 268 of 11,020 fund-quarters carry an amendment as well.
                 filing_date=('filing_date', 'min'),
                 net_assets=('net_assets', 'first'),
                 series_name=('series_name', 'first')).reset_index()
    cov = (H[H.seen].groupby(['series_id', 'q']).value.sum().rename('v_seen')
           .reset_index())
    base = base.merge(cov, on=['series_id', 'q'], how='left')
    base['labelled'] = 100 * base.v_seen.fillna(0) / base.port_value

    sub = H[H.seen].copy()
    for m in MEAS:
        col, ecol = 'm_' + m, 'e_' + m
        ok = sub[col].notna()
        a = (sub[ok].assign(_w=sub.loc[ok, 'value'] * sub.loc[ok, col])
             .groupby(['series_id', 'q'])
             .agg(_w=('_w', 'sum'), _v=('value', 'sum')).reset_index())
        a[m] = a._w / a._v
        base = base.merge(a[['series_id', 'q', m]], on=['series_id', 'q'], how='left')
        # abnormal, on the benchmark-covered sleeve only
        eb = ok & sub[ecol].notna()
        b = (sub[eb].assign(_a=sub.loc[eb, 'value'] * sub.loc[eb, col],
                            _e=sub.loc[eb, 'value'] * sub.loc[eb, ecol])
             .groupby(['series_id', 'q'])
             .agg(_a=('_a', 'sum'), _e=('_e', 'sum'), _v=('value', 'sum')).reset_index())
        b['abn_' + m] = (b._a - b._e) / b._v
        base = base.merge(b[['series_id', 'q', 'abn_' + m]], on=['series_id', 'q'],
                          how='left')

    # ---------------- the contemporaneous claim ----------------
    P = pd.read_parquet(PANEL)
    P['date'] = pd.to_datetime(P.date, errors='coerce')
    P = P[P.series_resolved & P.series_id.notna() & P.date.notna()]
    P = P[['series_id', 'date'] + [c for c in CLAIMS if c in P.columns]].sort_values('date')
    base['report_date'] = pd.to_datetime(base.report_date)
    base = base.sort_values('report_date')
    A = pd.merge_asof(base, P, left_on='report_date', right_on='date', by='series_id',
                      direction='backward')
    # a prospectus more than 16 months old is not the one in force (asof.py)
    A, n_expired = expire(A, 'report_date', [c for c in P.columns if c != 'series_id'])
    A['claim_age_days'] = (A.report_date - A.date).dt.days
    for c in CLAIMS:
        if c in A.columns:
            A[c] = A[c].astype('boolean').fillna(False).astype(bool)

    w('-' * 96)
    w('COVERAGE AND ALIGNMENT')
    w('-' * 96)
    w('  fund-quarters %d over %d funds' % (len(A), A.series_id.nunique()))
    w('  with a prospectus public by the report date : %.1f%% (median age %.0f days)'
      % (100 * A.date.notna().mean(), A.claim_age_days.median()))
    w('  ... matches older than %d days treated as unknown : %d fund-quarters'
      % (MAX_AGE_DAYS, n_expired))
    w('  %-14s %12s %12s %12s'
      % ('measure', 'cov median', 'raw median', 'abnormal med'))
    for m in MEAS:
        w('  %-14s %11.0f%% %12.1f %12.1f'
          % (m, A.labelled.median(), A[m].median(), A['abn_' + m].median()))
    w()
    w('  Abnormal medians sit near zero by construction - the benchmark is built from')
    w('  the same firms - so the level is uninformative and only the CROSS-SECTION is.')
    w()

    A.to_parquet(os.path.join(OUT, DEST), index=False)
    w('wrote out/%s' % DEST)

    with io.open(os.path.join(LOGS, 'ai_exposure.txt'), 'w', encoding='utf-8') as fh:
        fh.write(buf.getvalue())


if __name__ == '__main__':
    main()

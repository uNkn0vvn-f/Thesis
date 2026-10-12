"""Holdings for every fund in the test, with the identifiers that reach foreign listings.

Scope: the 946 series in the frozen identification layer, plus the 954 thematic funds
from the theme universe, so the comparison group is extracted in the same pass
and on the same terms. 1,696 series in all.

WHY ISIN AND LEI AND NOT JUST CUSIP. Bridging holdings to firms on
9-digit CUSIP alone, which only US listings carry, does measurable
damage: Japan, Taiwan, Korea and the Netherlands match at 0.0%, and the names that
fail are TSMC, Samsung, SK Hynix and ASML. Those are the AI hardware supply chain, so
the attrition is not random - it silently deletes exactly the `enabler` leg that this
test is about. N-PORT carries `ISSUER_LEI` on the holding and `IDENTIFIER_ISIN` keyed on
`HOLDING_ID`, and Compustat Global has ISIN, so the bridge can be built on gvkey instead.

Equity only (`ASSET_CAT == 'EC'`): the claims under test are about the companies a fund
owns, and a swap or an option on an AI index is a different question, handled by the
`is_overlay` axis in src/fund_claims/.
"""
import csv, glob, io, os, re, zipfile

import pandas as pd

from src import paths as tp  # noqa: E402

OUT = tp.out('holdings')
LOGS = tp.logs('holdings')
os.makedirs(OUT, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)

NPORT = tp.data('nport')
IDENT = tp.out('fund_claims', 'identification_fund.parquet')
THEME = tp.out('holdings', 'theme_universe.parquet')
DEST = os.path.join(OUT, 'holdings.parquet')
FROM_Q = os.environ.get('FROM_Q', '2021q1')
csv.field_size_limit(1 << 24)


def tsv(z, name):
    if name not in z.namelist():
        return [], iter(())
    t = io.TextIOWrapper(z.open(name), encoding='utf-8', errors='replace', newline='')
    rd = csv.reader(t, delimiter='\t')
    try:
        return next(rd), rd
    except StopIteration:
        return [], iter(())


def main():
    buf = io.StringIO()

    def w(s=''):
        buf.write(s + '\n')
        try:
            print(s)
        except UnicodeEncodeError:
            print(s.encode('ascii', 'replace').decode())

    F = pd.read_parquet(IDENT)
    TH = pd.read_parquet(THEME)
    ai_ids = set(F.index)
    th_ids = set(TH[TH.is_thematic].series_id)
    ids = ai_ids | th_ids

    w('=' * 96)
    w('HOLDINGS EXTRACTION')
    w('=' * 96)
    w('  identification universe : %d series' % len(ai_ids))
    w('  thematic comparison     : %d series' % len(th_ids))
    w('  union to extract        : %d series' % len(ids))
    w('  from %s, equity positions only' % FROM_Q)
    w()

    rows = []
    file_qs = []
    for p in sorted(glob.glob(os.path.join(NPORT, '*_nport.zip'))):
        q = re.search(r'(\d{4}q\d)', os.path.basename(p)).group(1)
        if q < FROM_Q:
            continue
        file_qs.append(q)
        z = zipfile.ZipFile(p)

        h, rd = tsv(z, 'SUBMISSION.tsv')
        i = {c: j for j, c in enumerate(h)}
        sub = {r[i['ACCESSION_NUMBER']]: (r[i['REPORT_DATE']], r[i['FILING_DATE']])
               for r in rd}

        h, rd = tsv(z, 'FUND_REPORTED_INFO.tsv')
        i = {c: j for j, c in enumerate(h)}
        meta = {}
        for r in rd:
            sid = r[i['SERIES_ID']].strip()
            if sid in ids:
                meta[r[i['ACCESSION_NUMBER']]] = (sid, r[i['SERIES_NAME']],
                                                  r[i['NET_ASSETS']])
        if not meta:
            continue

        # ISIN is keyed on HOLDING_ID in a separate table
        h, rd = tsv(z, 'IDENTIFIERS.tsv')
        i = {c: j for j, c in enumerate(h)}
        isin = {}
        if h:
            for r in rd:
                v = r[i['IDENTIFIER_ISIN']].strip()
                if v:
                    isin[r[i['HOLDING_ID']]] = v

        h, rd = tsv(z, 'FUND_REPORTED_HOLDING.tsv')
        i = {c: j for j, c in enumerate(h)}
        n = 0
        for r in rd:
            a = r[i['ACCESSION_NUMBER']]
            if a not in meta:
                continue
            if r[i['ASSET_CAT']] != 'EC':
                continue
            hid = r[i['HOLDING_ID']]
            sid, snm, na = meta[a]
            rd_, fd_ = sub.get(a, ('', ''))
            rows.append((q, rd_, fd_, a, sid, snm, na, hid,
                         r[i['ISSUER_NAME']], r[i['ISSUER_LEI']],
                         r[i['ISSUER_CUSIP']], isin.get(hid, ''),
                         r[i['CURRENCY_VALUE']], r[i['PERCENTAGE']],
                         r[i['INVESTMENT_COUNTRY']], r[i['PAYOFF_PROFILE']]))
            n += 1
        w('  %s  %5d equity rows over %d funds'
          % (q, n, len(set(m[0] for m in meta.values()))))

    H = pd.DataFrame(rows, columns=[
        'file_q', 'report_date', 'filing_date', 'accession', 'series_id', 'series_name',
        'net_assets', 'holding_id', 'issuer_name', 'issuer_lei', 'cusip', 'isin',
        'value', 'pct', 'country', 'payoff'])
    for c in ('net_assets', 'value', 'pct'):
        H[c] = pd.to_numeric(H[c], errors='coerce')
    H['report_date'] = pd.to_datetime(H.report_date, format='%d-%b-%Y', errors='coerce')
    H['filing_date'] = pd.to_datetime(H.filing_date, format='%d-%b-%Y', errors='coerce')

    # ---- the portfolio's own quarter, one filing per portfolio date, complete quarters ----
    n0 = len(H)
    H = H[H.report_date.notna()].copy()
    n_nodate = n0 - len(H)
    H['q'] = (H.report_date.dt.year.astype(str) + 'q'
              + H.report_date.dt.quarter.astype(str))
    last_file = max(file_qs)
    y, k = int(last_file[:4]), int(last_file[-1])
    last_ok = '%dq%d' % ((y, k - 1) if k > 1 else (y - 1, 4))
    F0 = (H[['series_id', 'report_date', 'filing_date', 'accession']].drop_duplicates()
          .sort_values(['series_id', 'report_date', 'filing_date', 'accession']))
    keep = F0.groupby(['series_id', 'report_date']).tail(1).accession
    n_acc0 = F0.accession.nunique()
    H = H[H.accession.isin(set(keep))]
    n_superseded = n_acc0 - H.accession.nunique()
    in_win = (H.q >= FROM_Q) & (H.q <= last_ok)
    n_outside = H[~in_win].groupby(['series_id', 'q']).ngroups
    H = H[in_win].copy()
    # a fund that moves its fiscal year can report twice inside one calendar quarter;
    # the later portfolio stands for the quarter, so no cell mixes two dates
    last_rd = H.groupby(['series_id', 'q']).report_date.transform('max')
    n_same_q = H[H.report_date < last_rd].accession.nunique()
    H = H[H.report_date == last_rd].copy()
    H = H[['q'] + [c for c in H.columns if c != 'q']]

    # Persist BEFORE reporting. A crash in the summary section once destroyed a ten
    # minute extraction; the expensive artefact is saved first now.
    H.to_parquet(DEST, index=False)

    w()
    w('-' * 96)
    w('WHAT CAME OUT')
    w('-' * 96)
    w('  equity position rows : %d' % len(H))
    w('  funds                : %d  (of %d requested)'
      % (H.series_id.nunique(), len(ids)))
    w('    in the identification universe : %d'
      % H[H.series_id.isin(ai_ids)].series_id.nunique())
    w('    thematic comparison only       : %d'
      % H[H.series_id.isin(th_ids - ai_ids)].series_id.nunique())
    w('  fund-quarters        : %d' % H.groupby(['series_id', 'q']).ngroups)
    w('  quarters             : %s .. %s  (portfolio date; data sets %s .. %s)'
      % (H.q.min(), H.q.max(), min(file_qs), last_file))
    w('  rows without a portfolio date, dropped        : %d' % n_nodate)
    w('  filings superseded by a later one, same date  : %d' % n_superseded)
    w('  fund-quarters outside %s .. %s, dropped   : %d  (%s is not complete yet)'
      % (FROM_Q, last_ok, n_outside, last_file))
    w('  earlier portfolio in the same quarter, dropped: %d filings' % n_same_q)
    w('  fund-quarters holding more than one filing    : %d  (must be 0)'
      % int((H.groupby(['series_id', 'q']).accession.nunique() > 1).sum()))
    w()
    w('  identifier coverage, by row and by value:')
    tot = H.value.sum()
    for nm, m in (('CUSIP present', H['cusip'].str.strip().ne('') & H['cusip'].ne('N/A')),
                  ('ISIN present', H['isin'].str.strip().ne('')),
                  ('LEI present', H['issuer_lei'].str.strip().ne('')
                   & H['issuer_lei'].ne('N/A')),
                  ('CUSIP or ISIN', (H['cusip'].str.strip().ne('') & H['cusip'].ne('N/A'))
                   | H['isin'].str.strip().ne(''))):
        w('    %-16s %6.1f%% of rows   %6.1f%% of value'
          % (nm, 100 * m.mean(), 100 * H.loc[m, 'value'].sum() / tot))
    w()
    w('  top investment countries by value:')
    cv = H.groupby('country').value.sum().sort_values(ascending=False)
    for c, v in cv.head(10).items():
        sub = H[H.country == c]
        has_isin = sub['isin'].str.strip().ne('').mean()
        has_cusip = (sub['cusip'].str.strip().ne('') & sub['cusip'].ne('N/A')).mean()
        w('    %-6s %6.2f%% of value   ISIN %5.1f%%   CUSIP %5.1f%%'
          % (c or '(blank)', 100 * v / tot, 100 * has_isin, 100 * has_cusip))
    w()
    w('  This is the table that decides whether the enabler leg is testable: if CUSIP')
    w('  were the only key, every foreign semiconductor holding would drop out.')

    w()
    w('wrote out/holdings.parquet (%d rows)' % len(H))

    with io.open(os.path.join(LOGS, 'h01_extract_holdings.txt'), 'w',
                 encoding='utf-8') as fh:
        fh.write(buf.getvalue())


if __name__ == '__main__':
    main()

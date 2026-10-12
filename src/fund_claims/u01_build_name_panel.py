"""A point-in-time name panel for every registered fund, from two independent sources.

Taking each series' name from its FIRST filing and holding it fixed
(nm=('nm','first')) has a cost: seven funds that renamed INTO an AI name never enter the sample at all, one
that renamed OUT of one stays in for its whole life, and Global X Artificial Intelligence
& Technology ETF - among the largest AI-themed ETFs in existence - is missing entirely
because in 2020 it was called Global X Future Analytics Tech ETF.

Sources, deliberately two:
  N-PORT   FUND_REPORTED_INFO.SERIES_NAME, one observation per fund per quarterly
           filing, near-complete coverage of registered funds
  485BPOS  series_name per prospectus filing, annual, but it is the record that links
           to the strategy text the classifier reads

Output is series_id x observation_date with the name as filed on that date. Nothing is
flagged AI here; that is handled by src/fund_claims/u02_ai_name_spells.py.
"""
import csv, glob, io, json, os, re, sys, zipfile
from datetime import datetime

import pandas as pd

from src import paths as tp  # noqa: E402

OUT = tp.out('fund_claims')
LOGS = tp.logs('fund_claims')
os.makedirs(OUT, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)

NPORT = tp.data('nport')
CSV485 = tp.out('disclosure', 'fund_classifications_2020_2026.csv')
csv.field_size_limit(1 << 24)


def tsv(z, name):
    """The header and a row reader of one TSV inside an N-PORT ZIP; empty if absent."""
    if name not in z.namelist():
        return [], iter(())
    t = io.TextIOWrapper(z.open(name), encoding='utf-8', errors='replace', newline='')
    rd = csv.reader(t, delimiter='\t')
    try:
        return next(rd), rd
    except StopIteration:
        return [], iter(())


def parse_dt(s):
    """A date in either N-PORT format, or None."""
    for f in ('%d-%b-%Y', '%Y-%m-%d'):
        try:
            return datetime.strptime(s.strip(), f)
        except Exception:
            pass
    return None


def from_nport():
    """Fund names from every N-PORT data set: series, name, report and filing dates,
    registrant, LEI and net assets."""
    rows = []
    for p in sorted(glob.glob(os.path.join(NPORT, '*_nport.zip'))):
        q = re.search(r'(\d{4}q\d)', os.path.basename(p)).group(1)
        z = zipfile.ZipFile(p)

        h, rd = tsv(z, 'SUBMISSION.tsv')
        i = {c: j for j, c in enumerate(h)}
        sub = {}
        for r in rd:
            sub[r[i['ACCESSION_NUMBER']]] = (parse_dt(r[i['REPORT_DATE']]),
                                             parse_dt(r[i['FILING_DATE']]))

        h, rd = tsv(z, 'REGISTRANT.tsv')
        i = {c: j for j, c in enumerate(h)}
        reg = {r[i['ACCESSION_NUMBER']]: (r[i['CIK']], r[i['REGISTRANT_NAME']])
               for r in rd}

        h, rd = tsv(z, 'FUND_REPORTED_INFO.tsv')
        i = {c: j for j, c in enumerate(h)}
        need = ('ACCESSION_NUMBER', 'SERIES_ID', 'SERIES_NAME', 'SERIES_LEI', 'NET_ASSETS')
        for c in need:
            if c not in i:
                raise KeyError('%s missing %s' % (p, c))
        n = 0
        for r in rd:
            a = r[i['ACCESSION_NUMBER']]
            sid = r[i['SERIES_ID']].strip()
            nm = r[i['SERIES_NAME']].strip()
            if not sid or not nm:
                continue
            rd_, fd_ = sub.get(a, (None, None))
            cik, rn = reg.get(a, ('', ''))
            rows.append((sid, nm, rd_, fd_, a, cik, rn, r[i['SERIES_LEI']].strip(),
                         r[i['NET_ASSETS']], q, 'nport'))
            n += 1
        print('  %s  %d fund rows' % (q, n), flush=True)
    return pd.DataFrame(rows, columns=['series_id', 'series_name', 'obs_date',
                                       'filing_date', 'accession', 'cik',
                                       'registrant_name', 'lei', 'net_assets',
                                       'quarter', 'src'])


def from_485():
    """Fund names from the classified 485BPOS filings, dated by filing date."""
    d = pd.read_csv(CSV485, encoding='utf-8-sig', low_memory=False,
                    usecols=['filing_date', 'accession', 'registrant_cik',
                             'registrant_name', 'series_id', 'series_name'])
    d['filing_date'] = pd.to_datetime(d.filing_date, errors='coerce')
    d = d[d.filing_date.notna() & d.series_id.notna() & d.series_name.notna()].copy()
    d['series_name'] = d.series_name.astype(str).str.strip()
    d['series_id'] = d.series_id.astype(str).str.strip()
    return pd.DataFrame({'series_id': d.series_id, 'series_name': d.series_name,
                         'obs_date': d.filing_date, 'filing_date': d.filing_date,
                         'accession': d.accession, 'cik': d.registrant_cik.astype(str),
                         'registrant_name': d.registrant_name, 'lei': '',
                         'net_assets': pd.NA, 'quarter': '', 'src': '485bpos'})


def main():
    """Combine both sources into the point-in-time name panel and report where they
    disagree."""
    buf = io.StringIO()

    def w(s=''):
        """Print a line and keep it for the log file."""
        buf.write(s + '\n')
        try:
            print(s)
        except UnicodeEncodeError:
            print(s.encode('ascii', 'replace').decode())

    # The N-PORT names are cached against the hashes of the ZIP files.
    cache = os.path.join(OUT, '_nport_names.parquet')
    zips = sorted(glob.glob(os.path.join(NPORT, '*_nport.zip')))
    if tp.cache_valid(cache, zips):
        N = pd.read_parquet(cache)
        print('reused %s' % cache)
    else:
        print('reading N-PORT ...')
        N = from_nport()
        N.to_parquet(cache, index=False)
        tp.cache_written(cache, zips)
    P = from_485()

    # One row per observation of a fund name in either source.
    A = pd.concat([N, P], ignore_index=True)
    A['obs_date'] = pd.to_datetime(A.obs_date, errors='coerce')
    A = A[A.obs_date.notna()].copy()
    A['series_name'] = A.series_name.astype(str).str.strip()
    A = A.sort_values(['series_id', 'obs_date'])

    w('=' * 96)
    w('POINT-IN-TIME NAME PANEL')
    w('=' * 96)
    w('  %-12s %10s %10s %14s %14s' % ('source', 'rows', 'series', 'first obs', 'last obs'))
    for s, v in A.groupby('src'):
        w('  %-12s %10d %10d %14s %14s'
          % (s, len(v), v.series_id.nunique(),
             str(v.obs_date.min().date()), str(v.obs_date.max().date())))
    w('  %-12s %10d %10d' % ('combined', len(A), A.series_id.nunique()))
    w()

    # Which series each source covers.
    ns = A.groupby('src').series_id.apply(set)
    only_n = ns['nport'] - ns['485bpos']
    only_p = ns['485bpos'] - ns['nport']
    w('  series in N-PORT only  : %d' % len(only_n))
    w('  series in 485BPOS only : %d' % len(only_p))
    w('  series in both         : %d' % len(ns['nport'] & ns['485bpos']))
    w()
    w('  Coverage matters when a universe uses 485BPOS alone. A fund')
    w('  that renames between prospectus filings is observed under the old name for')
    w('  up to a year; N-PORT catches the change within a quarter.')
    w()

    # ---- do the two sources agree on the name at the same time? ----
    w('-' * 96)
    w('DO THE TWO SOURCES AGREE ON THE NAME?')
    w('-' * 96)
    both = A[A.series_id.isin(ns['nport'] & ns['485bpos'])].copy()
    both['yr'] = both.obs_date.dt.year
    piv = both.groupby(['series_id', 'yr', 'src']).series_name.first().unstack()
    piv = piv.dropna()

    def norm(s):
        """A name reduced to lower-case letters and digits, for comparing the two sources."""
        return re.sub(r'[^a-z0-9]', '', str(s).lower())

    same = (piv['nport'].map(norm) == piv['485bpos'].map(norm))
    w('  series-years observed in both sources : %d' % len(piv))
    w('  identical after normalisation         : %.1f%%' % (100 * same.mean()))
    w()
    w('  a sample of same-year disagreements (these are real renames or')
    w('  registrant-style differences, and each one is a date the label could move):')
    for sid, r in piv[~same].head(12).iterrows():
        w('    %s %s' % (sid[0], sid[1]))
        w('        N-PORT   %s' % str(r['nport'])[:74])
        w('        485BPOS  %s' % str(r['485bpos'])[:74])
    w()

    A.to_parquet(os.path.join(OUT, 'name_panel.parquet'), index=False)
    w('wrote out/name_panel.parquet  (%d rows, %d series)' % (len(A), A.series_id.nunique()))

    with io.open(os.path.join(LOGS, 'u01_name_panel.txt'), 'w', encoding='utf-8') as fh:
        fh.write(buf.getvalue())


if __name__ == '__main__':
    main()

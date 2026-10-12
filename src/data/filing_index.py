"""The lists of 485BPOS filings to download, from EDGAR's quarterly form indexes.

For every quarter of the study, 2020Q1 to 2026Q3, the full-index file

    https://www.sec.gov/Archives/edgar/full-index/<year>/QTR<q>/form.idx

is read (downloaded once and kept as THESIS_DATA/edgar/form_<year>QTR<q>.idx) and its
485BPOS rows filed by LAST_DATE are written, in index order, to two lists in
THESIS_DATA/edgar/:

    485bpos_2020_2023.csv    2020Q1 to 2023Q4
    485bpos_2024_2026.csv    2024Q1 to 2026Q3

with the columns form, company, cik, date and path. src.data.download_485 and
src.data.fetch_headers read both lists. A row is one filer of a filing, so a filing with
co-registrants appears once per filer; the downloads keep one record per accession.

LAST_DATE is the last filing date of the study's corpus, the "Last Data Received" date of
the 2026Q3 index it was drawn from. The stored 2024-2026 list holds exactly the 2026Q3
filings of that index, but in two blocks, because they were added from two readings of the
quarter's index; the downloads do not depend on the order of the rows.

    python -m src.data.filing_index
"""
import csv
import io
import os
import re
import time

import requests

from src import paths as tp

EDGAR = tp.data('edgar')
LISTS = {'485bpos_2020_2023.csv': ((2020, 1), (2023, 4)),
         '485bpos_2024_2026.csv': ((2024, 1), (2026, 3))}
URL = 'https://www.sec.gov/Archives/edgar/full-index/%d/QTR%d/form.idx'
LINE = re.compile(r'^(\S+)\s\s+(.+?)\s\s+(\d+)\s\s+(\d{4}-\d{2}-\d{2})\s\s+(edgar/data/\S+)\s*$')
FIELDS = ['form', 'company', 'cik', 'date', 'path']
LAST_DATE = '2026-09-10'


def quarters(first, last):
    """Every (year, quarter) from `first` through `last`."""
    y, q = first
    while (y, q) <= last:
        yield y, q
        y, q = (y, q + 1) if q < 4 else (y + 1, 1)


def form_index(year, quarter):
    """The quarter's form.idx, from the local copy or from EDGAR."""
    path = os.path.join(EDGAR, 'form_%dQTR%d.idx' % (year, quarter))
    if not os.path.isfile(path):
        response = requests.get(URL % (year, quarter), timeout=300,
                                headers={'User-Agent': tp.sec_user_agent(),
                                         'Accept-Encoding': 'gzip, deflate'})
        response.raise_for_status()
        with open(path, 'wb') as fh:
            fh.write(response.content)
        time.sleep(0.2)
    with open(path, 'rb') as fh:
        return fh.read()


def parse(content):
    """The 485BPOS rows of a form index filed by LAST_DATE."""
    rows = []
    for line in io.TextIOWrapper(io.BytesIO(content), encoding='latin-1'):
        match = LINE.match(line.rstrip())
        if match and match.group(1) == '485BPOS' and match.group(4) <= LAST_DATE:
            rows.append(dict(zip(FIELDS, match.groups())))
    return rows


def main():
    """Write both filing lists from the quarterly indexes."""
    os.makedirs(EDGAR, exist_ok=True)
    for name, (first, last) in LISTS.items():
        rows = []
        for year, quarter in quarters(first, last):
            got = parse(form_index(year, quarter))
            if not got:
                raise SystemExit('%dQ%d: the form index lists no 485BPOS filing' % (year, quarter))
            rows += got
            print('%dQ%d  %5d rows' % (year, quarter, len(got)), flush=True)
        with open(os.path.join(EDGAR, name), 'w', encoding='utf-8', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print('wrote %s: %d rows' % (name, len(rows)), flush=True)


if __name__ == '__main__':
    main()

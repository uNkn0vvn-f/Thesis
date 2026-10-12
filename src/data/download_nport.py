"""Download the SEC's quarterly N-PORT (2020Q1 on) and N-CEN (2019Q3 on) data sets.

THESIS_DATA/nport/<yyyy>q<q>_nport.zip and THESIS_DATA/ncen/<yyyy>q<q>_ncen.zip. N-CEN
supplies the fund-type flags of the monthly return panel (src.analysis.flows.fund_month_panel).

Why the data sets rather than individual NPORT-P filings: one NPORT-P covers one
fund, so getting holdings fund-by-fund would mean thousands of requests and you
cannot tell which fund a filing belongs to without downloading it first. The DERA
quarterly ZIP carries every fund's holdings for the quarter in documented TSVs,
keyed by SERIES_ID - the same identifier the 485BPOS panel already uses.

Join path:
    our series_id -> FUND_REPORTED_INFO.SERIES_ID -> ACCESSION_NUMBER
                  -> FUND_REPORTED_HOLDING / REGISTRANT / SUBMISSION

The ZIPs are kept so any later change of extraction scope is a local operation.
"""
import os
import time

import requests

from src import paths as tp  # noqa: E402

END_YEAR = int(os.environ.get('END_YEAR', '2026'))
# data set: (destination, first quarter, URL); a quarter not yet released answers 404
SETS = {
    'nport': (tp.data('nport'), (int(os.environ.get('START_YEAR', '2020')), 1),
              'https://www.sec.gov/files/dera/data/form-n-port-data-sets/%dq%d_nport.zip'),
    'ncen': (tp.data('ncen'), (2019, 3),
             'https://www.sec.gov/files/dera/data/form-n-cen-data-sets/%dq%d_ncen.zip'),
}
H = {'User-Agent': tp.sec_user_agent(),
     'Accept-Encoding': 'gzip, deflate', 'Host': 'www.sec.gov'}


def download(kind):
    dest, (y0, q0), base = SETS[kind]
    os.makedirs(dest, exist_ok=True)
    got = miss = total = 0
    for y in range(y0, END_YEAR + 1):
        for q in (1, 2, 3, 4):
            if (y, q) < (y0, q0):
                continue
            name = '%dq%d_%s.zip' % (y, q, kind)
            p = os.path.join(dest, name)
            if os.path.exists(p) and os.path.getsize(p) > 1_000_000:
                total += os.path.getsize(p); got += 1
                print('  cached  %-18s %.0f MB' % (name, os.path.getsize(p) / 1e6), flush=True)
                continue
            try:
                t0 = time.time()
                with requests.get(base % (y, q), headers=H, timeout=1200, stream=True) as r:
                    if r.status_code != 200:
                        miss += 1
                        print('  missing %-18s HTTP %d (not yet released)' % (name, r.status_code), flush=True)
                        continue
                    tmp = p + '.part'
                    with open(tmp, 'wb') as fh:
                        for chunk in r.iter_content(1 << 20):
                            fh.write(chunk)
                os.replace(tmp, p)
                sz = os.path.getsize(p)
                total += sz; got += 1
                print('  fetched %-18s %.0f MB in %.0f s' % (name, sz / 1e6, time.time() - t0), flush=True)
            except Exception as e:
                miss += 1
                print('  ERROR   %-18s %r' % (name, e), flush=True)
            time.sleep(0.4)
    print('%s: quarters on disk %d, unavailable %d, %.1f GB\n' % (kind, got, miss, total / 1e9))


if __name__ == '__main__':
    for kind in SETS:
        download(kind)

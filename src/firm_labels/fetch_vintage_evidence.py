"""Fetch the filing text of the annual company-label vintages from SEC EDGAR.

The vintage for calendar year Y reads only filings public by 31 December of Y-1. For each
vintage this runs, in order:

    p01_fetch_pre    Item 1 (Business) of the last 10-K each company filed in Y-1, and
                     the vintage's accession list
    p05_fetch_20f    Item 4 (Information on the Company) of the last 20-F, or the annual
                     information form of the last 40-F, filed by Y-1 for companies that
                     have no 10-K text; it re-extends the accession list, which p01 rewrites

after p04_foreign_index has indexed the annual reports every universe company files with
the SEC (p05 picks its filings from that index). All three skip what is already on disk,
so a rerun on a complete evidence folder downloads nothing and leaves the stored text
unchanged. The patent profiles and as-of financials of the same vintages, the two
classifier runs and the consensus label are steps of `python run.py prepare`.

    python -m src.firm_labels.fetch_vintage_evidence            every vintage
    python -m src.firm_labels.fetch_vintage_evidence 2023 2024  selected vintages

Requests carry the User-Agent of src.paths.sec_user_agent().
"""
import os
import subprocess
import sys
import time

from src import paths as tp
from src.preparation import YEARS

VINTAGES = tp.out('firm_labels', 'vintages', '00_evidence')
WORKERS = os.environ.get('WORKERS', '6')


def run(label, module, env, log):
    """Run one fetch module, write its output to `log`, and stop on failure."""
    started = time.time()
    with open(log, 'w', encoding='utf-8') as stream:
        process = subprocess.run([sys.executable, '-m', module], env=dict(os.environ, **env),
                                 cwd=tp.REPO, stdout=stream, stderr=subprocess.STDOUT)
    print('    %-30s %s %5.0fs' % (label, 'ok' if process.returncode == 0 else 'failed',
                                   time.time() - started), flush=True)
    if process.returncode:
        with open(log, encoding='utf-8', errors='replace') as stream:
            print(stream.read()[-2500:])
        raise SystemExit('%s failed; see %s' % (label, log))


def main(argv=None):
    """Index the annual reports once, then fetch the 10-K and the 20-F or 40-F text of each
    vintage."""
    args = sys.argv[1:] if argv is None else argv
    years = [int(a) for a in args if a.isdigit()] or list(YEARS)
    os.makedirs(tp.logs('firm_labels'), exist_ok=True)
    print('annual report index')
    run('p04_foreign_index', 'src.firm_labels.p04_foreign_index', {},
        os.path.join(tp.logs('firm_labels'), 'p04_foreign_index.txt'))
    for year in years:
        prior = year - 1
        evidence = os.path.join(VINTAGES, str(year))
        os.makedirs(evidence, exist_ok=True)
        print('vintage %d (filings public by %d-12-31)' % (year, prior))
        run('10-K, Item 1', 'src.firm_labels.p01_fetch_pre',
            {'LM_LO': '%d0101' % prior, 'LM_HI': '%d1231' % prior, 'EVID_OUT': evidence,
             'WORKERS': WORKERS},
            os.path.join(evidence, 'fetch.txt'))
        run('20-F / 40-F', 'src.firm_labels.p05_fetch_20f',
            {'LM_HI': '%d1231' % prior, 'EVID_OUT': evidence, 'WORKERS': WORKERS},
            os.path.join(evidence, 'fetch_20f.txt'))


if __name__ == '__main__':
    main()

"""Prepare claims, holdings and firm labels from local source data and cached answers.

Run from the repository root with ``python run.py prepare``. Acquisition modules in
``src.data`` and the filing fetch of the company-label vintages
(``src.firm_labels.fetch_vintage_evidence``) run separately. Model requests are disabled
unless ``--allow-api`` is supplied.
"""
import argparse
import os
import subprocess
import sys
import time

from src import paths as tp


YEARS = range(2020, 2027)


def py(module, **env):
    """A Python module command and its per-step environment."""
    return [sys.executable, '-m', module], env


def steps():
    """Return processing steps in their data-dependency order."""
    fc, ho, fl = 'src.fund_claims.', 'src.holdings.', 'src.firm_labels.'
    sequence = [('classify disclosure', py('src.disclosure.classify_excerpts'))]
    sequence += [(name, py(fc + name)) for name in (
        'u01_build_name_panel', 'u02_ai_name_spells', 'c01_claim_type',
        's00_attribute', 's01_extract_evidence')]
    sequence += [('s02_classify run ' + run, py(fc + 's02_classify', RUN=run))
                 for run in ('1', '2')]
    sequence += [(name, py(fc + name)) for name in (
        's03_compose', 's04_final_labels', 'z01_freeze', 'a01_build_panel')]
    sequence += [(name, py(ho + name)) for name in (
        'theme_universe', 'v01_build_measures', 'h01_extract_holdings',
        'h02_bridge_to_gvkey', 'b01_stock_panel', 'c01_portfolio_exposure')]
    sequence += [('u01_build_universe', py(fl + 'u01_build_universe'))]
    vintages = tp.out('firm_labels', 'vintages')
    for year in YEARS:
        evidence, classification, labels = (
            os.path.join(vintages, stage, str(year))
            for stage in ('00_evidence', '01_classify', '02_label'))
        sequence += [
            ('p02_patent_pre %d' % year,
             py(fl + 'p02_patent_pre', UNTIL=str(year - 1), EVID_OUT=evidence)),
            ('p03_asof_financials %d' % year,
             py(fl + 'p03_asof_financials', CUTOFF='%d-12-31' % (year - 1), FIN_OUT=evidence)),
        ]
        sequence += [
            ('c01_classify %d run %s' % (year, run),
             py(fl + 'c01_classify', RUN=run, EVIDENCE='1', EVID=evidence,
                CLS_OUT=classification, FIN_ASOF=os.path.join(evidence, 'financials_asof.parquet')))
            for run in ('3', '4')]
        sequence += [
            ('p03_label_pre %d' % year,
             py(fl + 'p03_label_pre', CLS_DIR=classification, LAB_OUT=labels, EVID=evidence))]
    sequence += [('ai_exposure', py(ho + 'ai_exposure'))]
    # monthly fund flows and returns from the N-PORT and N-CEN data sets, and the base factors
    sequence += [(name, py('src.analysis.flows.' + name)) for name in ('fund_month_panel', 'base_factors')]
    return sequence


def select_start(sequence, start):
    """An exact step name wins; otherwise require an unambiguous substring."""
    if start is None:
        return 0
    exact = [index for index, (name, _) in enumerate(sequence) if name == start]
    matches = exact or [index for index, (name, _) in enumerate(sequence) if start in name]
    if not matches:
        raise ValueError('No step matches --from %r. Use --list-steps.' % start)
    if len(matches) != 1:
        raise ValueError('Ambiguous --from %r; matches: %s' %
                         (start, ', '.join(sequence[index][0] for index in matches)))
    return matches[0]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--from', dest='start', metavar='STEP',
                        help='exact step name or an unambiguous substring')
    parser.add_argument('--allow-api', action='store_true',
                        help='allow uncached classifications to call DeepSeek')
    parser.add_argument('--list-steps', action='store_true', help='list steps without running them')
    args = parser.parse_args(argv)
    sequence = steps()
    if args.list_steps:
        for index, (name, _) in enumerate(sequence):
            print('%02d %s' % (index, name))
        return
    try:
        first = select_start(sequence, args.start)
    except ValueError as error:
        parser.error(str(error))
    environment = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONHASHSEED='0', MPLBACKEND='Agg',
                       THESIS_NO_API='0' if args.allow_api else '1')
    log_dir = tp.logs('preparation')
    os.makedirs(log_dir, exist_ok=True)
    started = time.time()
    for index, (name, (command, env)) in enumerate(sequence[first:], first):
        step_started = time.time()
        log = os.path.join(log_dir, '%02d_%s.txt' % (index, name.replace(' ', '_')))
        with open(log, 'w', encoding='utf-8') as stream:
            process = subprocess.run(command, cwd=tp.REPO, env=dict(environment, **env),
                                     stdout=stream, stderr=subprocess.STDOUT)
        print('%02d %-28s %s %5.0fs' %
              (index, name, 'ok' if process.returncode == 0 else 'failed', time.time() - step_started),
              flush=True)
        if process.returncode:
            with open(log, encoding='utf-8', errors='replace') as stream:
                print(stream.read()[-1500:])
            raise SystemExit('Stopped at %s; see %s' % (name, os.path.relpath(log, tp.REPO)))
    print('Preparation completed in %.0f min.' % ((time.time() - started) / 60))


if __name__ == '__main__':
    main()

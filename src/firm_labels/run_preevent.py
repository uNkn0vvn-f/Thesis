"""Build the event analysis's pre-event company labels, one information set per event.

The ChatGPT and DeepSeek event designs weight each fund's pre-event holdings by company
AI-revenue labels. Those labels must not see the event, so each one is built from an
information set closed before it: the last 10-K, 20-F or 40-F each company filed in the
window ending the day before the event, patents filed through the event's year (ChatGPT) or
the year before it (DeepSeek), and the last revenue and market capitalisation public by the
cutoff (fiscal period end plus 90 days). The companies are the 3,006 of the classified universe
when the information sets were built, stored with them as firm_universe_preevent.parquet;
src.firm_labels.u01_build_universe now returns a slightly different universe, because the
holdings data were extended afterwards.

    event      10-K window                 patents filed through   financials public by
    chatgpt    2022-01-01 .. 2022-11-30    2022                    2022-11-29
    deepseek   2024-01-01 .. 2025-01-26    2024                    2025-01-26

Steps, each the pipeline's own module pointed at the event's folders through environment
settings, as `python run.py prepare` does for the annual vintages:

    fetch       p04_foreign_index, p01_fetch_pre and p05_fetch_20f  -> 00_evidence/<event>/
    patents     p02_patent_pre
    financials  p03_asof_financials
    classify    c01_classify, runs 3 and 4 (DeepSeek; needs the API or its cached responses)
    label       p03_label_pre, the consensus of the two runs      -> 02_label/out/<event>/
    exposure    src.holdings.ai_exposure on that label              -> 03_exposure/out/<event>/

The GPT-6 Luna labels on the same information sets come from
src/analysis/robustness/cross_model_labels.py and luna_measures.py.

    python -m src.firm_labels.run_preevent                       both events, all steps
    python -m src.firm_labels.run_preevent chatgpt --steps patents,financials
"""
import argparse
import os
import subprocess
import sys
import time

from src import paths as tp

EVENTS = {'chatgpt': {'lo': '20220101', 'hi': '20221130', 'until': '2022', 'cutoff': '2022-11-29'},
          'deepseek': {'lo': '20240101', 'hi': '20250126', 'until': '2024', 'cutoff': '2025-01-26'}}
STEPS = ['fetch', 'patents', 'financials', 'classify', 'label', 'exposure']
PRE = tp.out('analysis', 'preevent')
UNIVERSE = os.path.join(PRE, 'firm_universe_preevent.parquet')


def run(label, module, env, log):
    """Run one pipeline module with the event's settings, log its output, and stop on
    failure."""
    e = dict(os.environ, UNIVERSE=UNIVERSE, **env)
    t0 = time.time()
    with open(log, 'w', encoding='utf-8') as fh:
        p = subprocess.run([sys.executable, '-m', module], env=e, cwd=tp.REPO, stdout=fh,
                           stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
    print('    %-44s %s %6.0fs' % (label, 'OK  ' if p.returncode == 0 else 'FAIL', time.time() - t0))
    sys.stdout.flush()
    if p.returncode:
        print(open(log, encoding='utf-8', errors='replace').read()[-3000:])
        sys.exit(1)


def build(tag, steps):
    """Build the information set, labels and exposures of one event, for the requested
    steps."""
    w = EVENTS[tag]
    EVID = os.path.join(PRE, '00_evidence', tag)
    CLS = os.path.join(PRE, '01_classify', 'out', tag)
    LAB = os.path.join(PRE, '02_label', 'out', tag)
    EXP = os.path.join(PRE, '03_exposure', 'out', tag)
    for d in (EVID, CLS, LAB, EXP):
        os.makedirs(d, exist_ok=True)
    print('  %s  (filings %s..%s, patents through %s, financials public by %s)'
          % (tag, w['lo'], w['hi'], w['until'], w['cutoff']))
    workers = os.environ.get('WORKERS', '8')
    if 'fetch' in steps:
        run('index of annual reports (EDGAR)', 'src.firm_labels.p04_foreign_index', {},
            os.path.join(EVID, 'index.txt'))
        run('fetch 10-K filed %s..%s' % (w['lo'], w['hi']), 'src.firm_labels.p01_fetch_pre',
            {'LM_LO': w['lo'], 'LM_HI': w['hi'], 'EVID_OUT': EVID, 'WORKERS': workers},
            os.path.join(EVID, 'fetch.txt'))
        run('fetch 20-F and 40-F filed by %s' % w['hi'], 'src.firm_labels.p05_fetch_20f',
            {'LM_HI': w['hi'], 'EVID_OUT': EVID, 'WORKERS': workers},
            os.path.join(EVID, 'fetch_20f.txt'))
    if 'patents' in steps:
        run('patents filed through %s' % w['until'], 'src.firm_labels.p02_patent_pre',
            {'UNTIL': w['until'], 'EVID_OUT': EVID}, os.path.join(EVID, 'patent_run.txt'))
    if 'financials' in steps:
        run('financials public by %s' % w['cutoff'], 'src.firm_labels.p03_asof_financials',
            {'CUTOFF': w['cutoff'], 'FIN_OUT': EVID}, os.path.join(EVID, 'financials_run.txt'))
    if 'classify' in steps:
        for run_id in ('3', '4'):
            run('classify run %s' % run_id, 'src.firm_labels.c01_classify',
                {'EVIDENCE': '1', 'RUN': run_id, 'EVID': EVID, 'CLS_OUT': CLS,
                 'FIN_ASOF': os.path.join(EVID, 'financials_asof.parquet'), 'WORKERS': workers},
                os.path.join(CLS, 'run%s.txt' % run_id))
    if 'label' in steps:
        run('label (consensus of runs 3 and 4)', 'src.firm_labels.p03_label_pre',
            {'CLS_DIR': CLS, 'LAB_OUT': LAB, 'EVID': EVID}, os.path.join(LAB, 'label_run.txt'))
    if 'exposure' in steps:
        run('fund-quarter exposure on the pre-event label', 'src.holdings.ai_exposure',
            {'LAB': os.path.join(LAB, 'firm_ai_labels.parquet'), 'EXP_OUT': EXP},
            os.path.join(EXP, 'exposure_run.txt'))


def main(argv=None):
    """Parse the events and steps and build each event."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('events', nargs='*', help='chatgpt, deepseek (default: both)')
    parser.add_argument('--steps', default=','.join(STEPS), help='comma-separated subset of: ' + ', '.join(STEPS))
    args = parser.parse_args(argv)
    events = args.events or list(EVENTS)
    steps = [s.strip() for s in args.steps.split(',') if s.strip()]
    unknown = [s for s in steps if s not in STEPS] + [e for e in events if e not in EVENTS]
    if unknown:
        parser.error('unknown event or step: %s' % ', '.join(unknown))
    if not os.path.exists(UNIVERSE):
        sys.exit('missing %s, the event analysis\'s company universe' % UNIVERSE)
    for tag in events:
        build(tag, steps)


if __name__ == '__main__':
    main()

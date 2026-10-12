"""Run the robustness and inference analyses in dependency order and record input hashes.

    python src/analysis/robustness/run_all.py                      default analyses
    python src/analysis/robustness/run_all.py event_extensions figures   selected stages
    python src/analysis/robustness/run_all.py --check-inputs       preflight only
    python src/analysis/robustness/run_all.py validation_sample   optional: the agreement sample

No downloads or model calls. The label-dependent stages run twice: on the DeepSeek labels
(tables in results/tables/robustness/deepseek/) and on the GPT-6 Luna labels, the thesis
measurement (results/tables/robustness/). luna_measures.py first builds the Luna tables from
the cached responses of cross_model_labels.py. Figures go to results/figures/ and each
invocation has its own manifest and logs in data/interim/robustness/runs/<run_id>/.
run_manifest.json is a snapshot of the latest invocation, never a merge of different runs.
The optional validation_sample stage draws the stratified sample of fund filings and
company-years whose design weights the agreement table of cross_model_labels.py uses; it is
excluded from the default workflow because the stored sample is an input of that table. The DeepSeek pre-event exposures are built by src.holdings.ai_exposure on the pre-event
labels (LAB and EXP_OUT settings), the Luna ones by luna_measures.py. The
figures stage also draws manuscript Figures 1 and 2 from the tables of `python run.py`, which
runs first. event_shift_share (the event design's shift-share inference, about 30 minutes per
label set) needs the CRSP daily file data/raw/wrds/crsp_dsf_v2_2020_2025.parquet
(`TABLES=crsp_dsf_v2_2020_2025 python -m src.data.pull_wrds`).

The stages read the pipeline's interim outputs (`python run.py prepare`), the processed
CSVs, the monthly flow and return panels (src.analysis.flows.fund_month_panel), the claim
ladder (src.analysis.flows.claim_ladder), the base factors and the AI factor
(src.analysis.flows.base_factors, ai_factor) and the cached model responses listed in INPUTS;
a full run stops before the first stage when any external input is missing. Selected stages
check their own dependencies; outputs promised by an earlier selected stage are checked
again immediately before use. --check-inputs checks availability, not successful execution.
Manifests record the source tree, package versions, safe configuration fields and per-stage
input hashes. Directories follow
src/paths.py (THESIS_DATA, THESIS_OUTPUT). The exit code is 0 only when every selected
stage succeeded; a failed stage ends the run with exit code 1 after the manifest is written,
and an unknown stage name is an error (exit code 2).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src import paths as tp  # noqa: E402

HERE = Path(__file__).resolve().parent
RAW, INTERIM, RESULTS = Path(tp.DATA), Path(tp.OUTPUT), Path(tp.RESULTS)
LOGS = INTERIM / 'robustness'
PER_LABELS = ['claim_text_screen', 'correspondence_inference', 'flow_contrast_sensitivity',
              'flow_contrast_age', 'flow_contrast_premium', 'plot_flow_contrast', 'alignment_flows',
              'event_pretrends', 'event_timing', 'event_extensions', 'event_shift_share']
# scripts outside this folder, and arguments for particular stages
SCRIPT = {'plot_flow_contrast': ROOT / 'src' / 'analysis' / 'plot_flow_contrast.py'}
ARGS = {'luna_measures': ['--check-ladder']}
ALL_STAGES = ([(s, 'deepseek') for s in PER_LABELS] + [('luna_measures', 'luna')]
          + [(s, 'luna') for s in PER_LABELS]
          + [(s, 'luna') for s in ('flow_panel_checks', 'validation_sample',
                                   'cross_model_downstream', 'figures', 'manuscript_tables')])
OPTIONAL_STAGES = {'validation_sample'}
STAGES = [(s, lab) for s, lab in ALL_STAGES if s not in OPTIONAL_STAGES]
# inputs whose hashes the manifest records; paths relative to the project root, resolved
# through the data, interim and results directories of src/paths.py
INPUTS = ['data/processed/fund_quarter_exposure.csv', 'data/processed/firm_label_vintages.csv',
          'data/processed/holdings_for_exposure.csv.gz', 'data/processed/claim_filings.csv',
          'data/processed/claim_denominator_fund_year.csv',
          'data/interim/cross_model/gpt-6-luna/fund.jsonl', 'data/interim/cross_model/gpt-6-luna/company.jsonl',
          'data/interim/cross_model/gpt-6-luna/company_preevent.jsonl',
          'data/interim/analysis/preevent/02_label/out/chatgpt/firm_ai_labels.parquet',
          'data/interim/analysis/preevent/02_label/out/deepseek/firm_ai_labels.parquet',
          'data/interim/exposure/fund_quarter_exposure.parquet',
          'data/interim/holdings/stock_quarter.parquet', 'data/interim/holdings/theme_universe.parquet',
          'data/interim/firm_labels/firm_universe.parquet',
          'data/interim/fund_claims/panel_fund_filing.parquet', 'data/interim/fund_claims/evidence.jsonl.gz',
          'data/interim/analysis/fund_flows/out/fund_month_flows.parquet',
          'data/interim/analysis/monthly_returns/out/fund_month_returns.parquet',
          'data/interim/analysis/factors/out/factors_monthly.parquet',
          'data/interim/analysis/flow_ladder/out/fund_month_ladder.parquet',
          'data/interim/analysis/preevent/03_exposure/out/chatgpt/fund_quarter_exposure.parquet',
          'data/interim/analysis/preevent/03_exposure/out/deepseek/fund_quarter_exposure.parquet',
          'data/raw/wrds/ff_daily.parquet', 'data/raw/wrds/crsp_dsf_v2_2020_2025.parquet',
          'data/raw/wrds/ccm_linktable.parquet', 'data/interim/holdings/holdings_bridged.parquet',
          # run.py's tables behind manuscript Figures 1 and 2
          'results/tables/claims_annual.csv', 'results/tables/claims_composition.csv',
          'results/tables/claims_value_chain.csv', 'results/tables/claimed_vs_held_quarterly.csv']
INPUTS += ['data/interim/analysis/alpha/out/factors.parquet',
           'data/interim/fund_claims/labels_final_filing.parquet',
           'data/interim/fund_claims/ai_name_spells.parquet',
           'data/processed/claim_denominator_recovery.csv']
VINTAGES = [f'data/interim/firm_labels/vintages/02_label/{y}/firm_ai_labels.parquet'
            for y in range(2020, 2027)]
INPUTS += VINTAGES

FLOW = 'data/interim/analysis/fund_flows/out/fund_month_flows.parquet'
RETURNS = 'data/interim/analysis/monthly_returns/out/fund_month_returns.parquet'
THEME = 'data/interim/holdings/theme_universe.parquet'
HOLDINGS = 'data/interim/holdings/holdings_bridged.parquet'
UNIVERSE = 'data/interim/firm_labels/firm_universe.parquet'
BUCKETS = 'data/interim/holdings/stock_quarter.parquet'
HOLDINGS_CSV = 'data/processed/holdings_for_exposure.csv.gz'
EVIDENCE = 'data/interim/fund_claims/evidence.jsonl.gz'
DENOMINATOR = 'data/processed/claim_denominator_fund_year.csv'
RECOVERY = 'data/processed/claim_denominator_recovery.csv'
SOURCES = 'data/processed/sources.json'
ATTRIBUTED = 'data/interim/fund_claims/funds_attributed/'
CORE = ['results/tables/' + name + '.csv' for name in
        ('claims_annual', 'claims_composition', 'claims_value_chain', 'claimed_vs_held_quarterly')]


def label_inputs(lab):
    suffix = '_luna' if lab == 'luna' else ''
    d = 'data/interim/cross_model/gpt-6-luna/derived/'
    return {
        'claims': f'data/processed/claim_filings{suffix}.csv',
        'labels': f'data/processed/firm_label_vintages{suffix}.csv',
        'panel': d + 'panel_fund_filing.parquet' if suffix else 'data/interim/fund_claims/panel_fund_filing.parquet',
        'exposure': d + 'fund_quarter_exposure.parquet' if suffix else 'data/interim/exposure/fund_quarter_exposure.parquet',
        'ladder': f'data/interim/analysis/flow_ladder{suffix}/out/fund_month_ladder.parquet',
    }


def preevent_inputs(lab, kind='exposure'):
    suffix = '_luna' if lab == 'luna' else ''
    stage, name = ('03_exposure', 'fund_quarter_exposure') if kind == 'exposure' else ('02_label', 'firm_ai_labels')
    return [f'data/interim/analysis/preevent{suffix}/{stage}/out/{tag}/{name}.parquet'
            for tag in ('chatgpt', 'deepseek')]


def table(lab, name):
    return 'results/tables/robustness/' + ('deepseek/' if lab == 'deepseek' else '') + name


# Outputs needed by subsequent stages, not an inventory of every diagnostic written.
TABLE_OUTPUTS = {
    'claim_text_screen': ['claim_filings_risk_screen.csv'],
    'correspondence_inference': ['correspondence_inference.csv', 'correspondence_quarterly_medians.csv'],
    'flow_contrast_sensitivity': ['flow_contrast_sensitivity.csv', 'flow_contrast_inside_theme.csv', 'flow_contrast_log.json'],
    'flow_contrast_age': ['flow_contrast_age.csv'],
    'flow_contrast_premium': ['flow_contrast_premium.csv'],
    'event_pretrends': ['event_pretrends.json'],
    'event_timing': ['event_timing.json'],
    'event_extensions': ['event_extensions.csv', 'event_fund_collapse.csv', 'event_date_scan.csv', 'event_extensions_log.json'],
    'event_shift_share': ['shift_share.json'],
    'cross_model_downstream': ['cross_model_correspondence.csv'],
}


def stage_outputs(stage, lab):
    if stage == 'luna_measures':
        return list(label_inputs('luna').values()) + preevent_inputs('luna') + preevent_inputs('luna', 'label')
    return [table(lab, name) for name in TABLE_OUTPUTS.get(stage, [])]


def stage_inputs(stage, lab):
    p, ds = label_inputs(lab), label_inputs('deepseek')
    event = [ds['ladder']] + preevent_inputs(lab)
    mapping = {
        'claim_text_screen': [p['claims'], p['panel'], p['exposure'], EVIDENCE, HOLDINGS, DENOMINATOR, RECOVERY, SOURCES],
        'correspondence_inference': [p['exposure'], p['panel'], p['labels'], HOLDINGS_CSV, BUCKETS,
                                     table(lab, 'claim_filings_risk_screen.csv')],
        'flow_contrast_sensitivity': [p['ladder'], p['exposure'], FLOW],
        'flow_contrast_age': [p['ladder'], FLOW],
        'flow_contrast_premium': [p['ladder'], THEME],
        'plot_flow_contrast': [table(lab, 'flow_contrast_premium.csv')],
        'alignment_flows': [p['exposure'], p['ladder'], RETURNS, 'data/interim/analysis/alpha/out/factors.parquet'],
        'event_pretrends': event + preevent_inputs(lab, 'label') + [HOLDINGS, UNIVERSE],
        'event_timing': event,
        'event_extensions': event + [p['exposure'], HOLDINGS_CSV, UNIVERSE, THEME, HOLDINGS, FLOW],
        'event_shift_share': event + preevent_inputs(lab, 'label') + [HOLDINGS, UNIVERSE,
                            'data/raw/wrds/crsp_dsf_v2_2020_2025.parquet', 'data/raw/wrds/ccm_linktable.parquet'],
        'luna_measures': list(ds.values()) + preevent_inputs('deepseek') + preevent_inputs('deepseek', 'label')
                        + VINTAGES + [HOLDINGS_CSV, HOLDINGS, BUCKETS, THEME, UNIVERSE, EVIDENCE, FLOW,
                          'data/interim/fund_claims/labels_final_filing.parquet',
                          'data/interim/fund_claims/ai_name_spells.parquet']
                        + [f'data/interim/cross_model/gpt-6-luna/{name}.jsonl'
                           for name in ('fund', 'company', 'company_preevent')],
        'flow_panel_checks': [FLOW, HOLDINGS, 'data/raw/wrds/ff_daily.parquet'],
        'validation_sample': [ds['claims'], ds['exposure'], EVIDENCE, HOLDINGS_CSV] + VINTAGES,
        'cross_model_downstream': [DENOMINATOR, HOLDINGS_CSV, BUCKETS]
                                 + [v for label in ('deepseek', 'luna') for k, v in label_inputs(label).items() if k != 'panel']
                                 + [v for label in ('deepseek', 'luna') for kind in ('label', 'exposure')
                                    for v in preevent_inputs(label, kind)],
        'figures': CORE + [p['claims'], table(lab, 'correspondence_quarterly_medians.csv'),
                          table(lab, 'correspondence_inference.csv'), table(lab, 'event_date_scan.csv'),
                          table(lab, 'cross_model_correspondence.csv')],
        'manuscript_tables': [table(lab, name) for name in (
            'correspondence_inference.csv', 'cross_model_correspondence.csv',
            'flow_contrast_sensitivity.csv', 'flow_contrast_age.csv', 'flow_contrast_inside_theme.csv',
            'flow_contrast_log.json', 'flow_contrast_premium.csv', 'event_extensions.csv',
            'event_fund_collapse.csv', 'event_timing.json', 'event_pretrends.json', 'event_extensions_log.json',
            'shift_share.json')] + [table('deepseek', name) for name in (
                'flow_contrast_sensitivity.csv', 'event_extensions.csv', 'shift_share.json')],
    }
    return sorted(set(mapping[stage]))


def resolve(rel: str) -> Path:
    for prefix, base in (('data/raw/', RAW), ('data/interim/', INTERIM), ('data/processed/', tp.PROCESSED),
                         ('results/', RESULTS)):
        if rel.startswith(prefix):
            return Path(base) / rel[len(prefix):]
    return ROOT / rel


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def attributed_inputs():
    """The attributed strategy shards named in the export inventory, plus any additional shards."""
    source = resolve(SOURCES)
    if not source.is_file():
        raise ValueError('missing attribution inventory: ' + SOURCES)
    inventory = json.loads(source.read_text(encoding='utf-8'))
    if not isinstance(inventory, dict):
        raise ValueError('attribution inventory must be an object: ' + SOURCES)
    prefix = 'fund_claims/funds_attributed/'
    expected = [k for k in inventory if k.startswith(prefix) and k.endswith('.jsonl.gz')]
    if not expected or any('..' in Path(k).parts for k in expected):
        raise ValueError('no valid attributed-shard inventory in ' + SOURCES)
    actual = [ATTRIBUTED + p.name for p in resolve(ATTRIBUTED).glob('*.jsonl.gz') if p.is_file()]
    return sorted(set(['data/interim/' + k for k in expected] + actual))


def requirements(stages):
    shards = attributed_inputs() if any(s == 'claim_text_screen' for s, _ in stages) else []
    result = {}
    for stage, lab in stages:
        files = stage_inputs(stage, lab)
        if stage == 'claim_text_screen':
            files += shards
        if stage == 'validation_sample':
            # Optional evidence changes the generated sheets; hash it when it is available.
            files += [f'data/interim/firm_labels/vintages/00_evidence/{year}/{name}'
                      for year in range(2020, 2027) for name in ('business_text.jsonl', 'patent_profile.parquet')
                      if resolve(f'data/interim/firm_labels/vintages/00_evidence/{year}/{name}').is_file()]
        result[(stage, lab)] = sorted(set(files))
    return result


def preflight(stages, required, full=False):
    """Only earlier selected producers can defer a check; cached unrelated outputs cannot."""
    produced, external, deferred = set(), set(INPUTS if full else []), set()
    for key in stages:
        for rel in required[key]:
            (deferred if rel in produced else external).add(rel)
        produced.update(stage_outputs(*key))
    missing = sorted(rel for rel in external if not resolve(rel).is_file())
    return sorted(external), sorted(deferred), missing


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def source_snapshot():
    files = [p for p in (ROOT / 'src').rglob('*') if p.is_file() and p.suffix.lower() in ('.py', '.r')]
    files += [ROOT / name for name in ('run.py', 'requirements.txt') if (ROOT / name).is_file()]
    digests = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(files)}
    digest = hashlib.sha256(json.dumps(digests, sort_keys=True).encode('utf-8')).hexdigest()
    return {'sha256': digest, 'files': digests}


def environment_snapshot():
    # Explicit allowlist: never serialize API keys, passwords or arbitrary environment values.
    names = ('THESIS_DATA', 'THESIS_OUTPUT', 'N_PERM', 'PYTHONHASHSEED', 'MPLBACKEND',
             'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'TAG', 'BAL', 'PERM')
    return {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
            'packages': dict(sorted((d.metadata['Name'], d.version) for d in metadata.distributions()
                                    if d.metadata.get('Name'))),
            'configuration': {k: os.environ[k] for k in names if k in os.environ},
            'forced_child_environment': {'THESIS_NO_API': '1', 'PYTHONIOENCODING': 'utf-8'},
            'paths': {'raw': str(RAW), 'interim': str(INTERIM), 'processed': str(tp.PROCESSED),
                      'results': str(RESULTS)}}


def preserve_previous():
    """Keep the previous manifest, unaltered, under runs/previous-<content hash>/."""
    previous = LOGS / 'run_manifest.json'
    if previous.is_file():
        archive = LOGS / 'runs' / ('previous-' + sha(previous)[:20])
        archive.mkdir(parents=True, exist_ok=True)
        if not (archive / 'manifest.json').exists():
            shutil.copyfile(previous, archive / 'manifest.json')


def save_manifest(log, run_dir):
    payload = json.dumps(log, indent=2, ensure_ascii=False) + '\n'
    for target in (run_dir / 'manifest.json', LOGS / 'run_manifest.json'):
        temporary = target.with_name(target.name + '.' + log['run_id'] + '.tmp')
        temporary.write_text(payload, encoding='utf-8')
        temporary.replace(target)


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--check-inputs', action='store_true', help='check dependencies without running stages or writing a manifest')
    parser.add_argument('stages', nargs='*', help='stage names; validation_sample is explicit-only')
    args = parser.parse_args(argv)
    known = {s for s, _ in ALL_STAGES}
    unknown = set(args.stages) - known
    if unknown:
        parser.error('unknown stage(s): %s; stages: %s' % (', '.join(sorted(unknown)), ', '.join(sorted(known))))
    stages = [(s, lab) for s, lab in ALL_STAGES if s in args.stages] if args.stages else STAGES
    preflight_error = None
    try:
        required = requirements(stages)
    except (OSError, ValueError, TypeError) as error:
        preflight_error = str(error)
        required = {}
    external, deferred, missing = ([], [], []) if preflight_error else preflight(stages, required, full=not args.stages)
    if args.check_inputs:
        if preflight_error:
            print('input preflight failed: %s' % preflight_error, file=sys.stderr)
            return 2
        if missing:
            print('missing inputs:\n  ' + '\n  '.join(missing), file=sys.stderr)
            return 2
        print('Input preflight passed: %d external files; %d files will be checked after their selected producer.'
              % (len(external), len(deferred)))
        print('No stages were run and no manifest was written.')
        return 0

    LOGS.mkdir(parents=True, exist_ok=True)
    preserve_previous()
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    run_dir = LOGS / 'runs' / run_id
    run_dir.mkdir(parents=True)
    log = {'schema_version': 2, 'run_id': run_id, 'started': utc_now(), 'finished': None,
           'scope': 'selected' if args.stages else 'default', 'status': 'preflight', 'ok': False,
           'invocation': [sys.executable, str(Path(__file__).resolve()), *argv],
           'selected_stages': [{'stage': s, 'labels': lab} for s, lab in stages],
           'source': source_snapshot(), 'environment': environment_snapshot(),
           'inputs': {rel: sha(resolve(rel)) for rel in external if resolve(rel).is_file()},
           'deferred_inputs': deferred, 'missing_inputs': missing, 'preflight_error': preflight_error, 'stages': []}
    save_manifest(log, run_dir)
    if preflight_error or missing:
        print('input preflight failed: ' + preflight_error if preflight_error else
              'missing inputs:\n  ' + '\n  '.join(missing), file=sys.stderr)
        log.update(status='missing_inputs', finished=utc_now())
        save_manifest(log, run_dir)
        return 2
    exit_code = 0
    try:
        for s, lab in stages:
            files = required[(s, lab)]
            missing = [rel for rel in files if not resolve(rel).is_file()]
            entry = {'stage': s, 'labels': lab, 'started': utc_now(), 'finished': None,
                     'status': 'missing_inputs' if missing else 'running', 'exit': None,
                     'inputs': {rel: sha(resolve(rel)) for rel in files if resolve(rel).is_file()},
                     'missing_inputs': missing}
            log['stages'].append(entry)
            if missing:
                print('%s (%s): missing inputs:\n  %s' % (s, lab, '\n  '.join(missing)), file=sys.stderr)
                entry.update(finished=utc_now(), exit=2)
                exit_code = 2
                break
            env = dict(os.environ, PYTHONIOENCODING='utf-8', THESIS_NO_API='1', THESIS_LABELS=lab)
            script = SCRIPT.get(s, HERE / (s + '.py'))
            command = [sys.executable, str(script)] + ARGS.get(s, [])
            entry['command'] = command
            log['status'] = 'running'
            save_manifest(log, run_dir)
            t0 = time.monotonic()
            r = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                               text=True, encoding='utf-8', errors='replace')
            log_name = '%s_%s.txt' % (s, lab)
            (run_dir / log_name).write_text(r.stdout + r.stderr, encoding='utf-8')
            entry.update(exit=r.returncode, seconds=round(time.monotonic() - t0, 1),
                         finished=utc_now(), status='succeeded' if r.returncode == 0 else 'failed', log=log_name,
                         outputs={rel: sha(resolve(rel)) for rel in stage_outputs(s, lab) if resolve(rel).is_file()})
            print(s, lab, 'OK' if r.returncode == 0 else 'FAILED', flush=True)
            save_manifest(log, run_dir)
            if r.returncode:
                exit_code = 1
                break
    except KeyboardInterrupt:
        log['error'] = 'Interrupted by user'
        if log['stages'] and log['stages'][-1]['status'] == 'running':
            log['stages'][-1].update(status='interrupted', finished=utc_now())
        exit_code = 130
    except OSError as error:
        log['error'] = str(error)
        if log['stages'] and log['stages'][-1]['status'] == 'running':
            log['stages'][-1].update(status='failed', error=str(error), finished=utc_now())
        exit_code = 1
    finally:
        log.update(finished=utc_now(), status='succeeded' if exit_code == 0 else 'failed', ok=exit_code == 0)
        log['source_at_finish'] = source_snapshot()['sha256']
        log['source_changed_during_run'] = log['source_at_finish'] != log['source']['sha256']
        save_manifest(log, run_dir)
    print('Run manifest: %s' % (run_dir / 'manifest.json'))
    return exit_code


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

"""Export analysis inputs from the intermediate data pipeline."""
import argparse
from datetime import date
import gzip
import hashlib
import io
import json
from pathlib import Path

import pandas as pd
from src import paths

def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == '.gz':
        with path.open('wb') as raw:
            with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as zipped:
                with io.TextIOWrapper(zipped, encoding='utf-8', newline='') as stream:
                    frame.to_csv(stream, index=False, lineterminator='\n', float_format='%.17g')
    else:
        frame.to_csv(path, index=False, lineterminator='\n', float_format='%.17g')


def resolve_series_id(row, header_series):
    """Direct attribution wins; an orphan is recovered only from a one-series header."""
    direct = (row.get('series_id') or '').strip()
    if direct:
        return direct, False
    series = header_series.get(row.get('accession'), ())
    if len(series) == 1 and series[0]:
        return series[0], True
    return None, False


def scan_denominator(source, headers, record):
    """Strict independent reconstruction, including the panel's header recovery rule."""
    header_files = sorted(headers.glob('*.jsonl.gz'))
    if not headers.is_dir() or not header_files:
        raise FileNotFoundError('No SEC header shards; supply --headers to verify series recovery')
    header_series, header_sources = {}, {}
    for shard in header_files:
        digest = record(shard, 'headers')
        with gzip.open(shard, 'rt', encoding='utf-8') as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                accession = row.get('accession')
                if not accession:
                    raise ValueError(f'Header lacks accession: {shard.name}:{line_number}')
                series = tuple((item.get('series_id') or '').strip()
                               for item in (row.get('series') or []))
                if accession in header_series and header_series[accession] != series:
                    raise ValueError(f'Conflicting series headers for {accession}')
                header_series[accession] = series
                header_sources.setdefault(accession, (f'raw_headers/{shard.name}',
                                                       digest, line_number))
    fund_years, direct_years, recovery = set(), set(), []
    shards = sorted((source / 'fund_claims/funds_attributed').glob('*.jsonl.gz'))
    if not shards:
        raise FileNotFoundError('No attributed prospectus shards; cannot freeze denominator')
    for shard in shards:
        digest = record(shard, 'output')
        with gzip.open(shard, 'rt', encoding='utf-8') as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                strategy = row.get('strategy') or ''
                if not strategy.strip():
                    continue
                filing_date = date.fromisoformat(str(row.get('date') or '')[:10])
                sid, recovered = resolve_series_id(row, header_series)
                if not sid:
                    continue
                key = (filing_date.year, sid)
                fund_years.add(key)
                if not recovered:
                    direct_years.add(key)
                    continue
                header_source, header_hash, header_line = header_sources[row['accession']]
                recovery.append({'year': filing_date.year, 'series_id': sid,
                                 'accession': row['accession'], 'filing_date': filing_date.isoformat(),
                                 'strategy_chars': len(strategy),
                                 'strategy_sha256': hashlib.sha256(strategy.encode('utf-8')).hexdigest(),
                                 'shard_source': shard.relative_to(source).as_posix(),
                                 'shard_sha256': digest, 'shard_line': line_number,
                                 'header_source': header_source, 'header_sha256': header_hash,
                                 'header_line': header_line, 'header_series_count': 1,
                                 'resolution_method': 'single_series_sec_header'})
    cols = ['year', 'series_id', 'accession', 'filing_date', 'strategy_chars',
            'strategy_sha256', 'shard_source', 'shard_sha256', 'shard_line',
            'header_source', 'header_sha256', 'header_line', 'header_series_count',
            'resolution_method', 'adds_fund_year_to_direct_denominator']
    for row in recovery:
        row['adds_fund_year_to_direct_denominator'] = (row['year'], row['series_id']) not in direct_years
    return (pd.DataFrame(sorted(fund_years), columns=['year', 'series_id']),
            pd.DataFrame(recovery, columns=cols),
            {'direct_fund_years': len(direct_years), 'resolved_fund_years': len(fund_years),
             'added_fund_years': len(fund_years - direct_years),
             'header_recovered_strategy_records': len(recovery), 'header_shards': len(header_files)})


def export(source, destination, headers):
    destination.mkdir(parents=True, exist_ok=True)
    sources = {}

    def record(path, kind=None):
        digest = sha256(path)
        key = ('raw_headers/' + path.relative_to(headers).as_posix()
               if kind == 'headers' else path.relative_to(source).as_posix())
        sources[key] = digest
        return digest if kind is not None else path

    def parquet(rel):
        return pd.read_parquet(record(source / rel))

    def save(frame, name, keys=None):
        if keys:
            frame = frame.sort_values(keys, kind='stable', na_position='last')
        write_csv(frame, destination / name)

    # Missing raw headers must be discovered before updating processed tables.
    denominator, recovery, _ = scan_denominator(source, headers, record)
    panel = parquet('fund_claims/panel_fund_filing.parquet')
    eligible = panel[panel.series_resolved & panel.series_id.notna() & panel.strategy_chars.gt(0)].copy()
    eligible['year'] = pd.to_datetime(eligible.date, errors='raise').dt.year
    expected = set(zip(eligible.year.astype(int), eligible.series_id))
    actual = set(zip(denominator.year, denominator.series_id))
    if actual != expected:
        raise ValueError('Resolved filing population differs from strategy/header records')

    save(parquet('fund_claims/identification_filing.parquet'), 'claim_filings.csv', ['date', 'series_id', 'accession'])
    save(parquet('fund_claims/flag_grades.parquet'), 'fund_label_agreement.csv')
    save(denominator, 'claim_denominator_fund_year.csv')
    save(recovery, 'claim_denominator_recovery.csv', ['year', 'series_id', 'accession', 'shard_source', 'shard_line'])
    exposure = parquet('exposure/fund_quarter_exposure.parquet')
    fields = ['series_id', 'q', 'n_holdings', 'port_value', 'v_seen', 'report_date',
              'filing_date', 'date', 'claim_age_days', 'labelled', 'uses_ai', 'theme_broad',
              'chain_developer', 'chain_enabler', 'chain_adopter', 'developer', 'enabler', 'adopter']
    save(exposure[fields], 'fund_quarter_exposure.csv', ['series_id', 'q'])
    holdings = parquet('holdings/holdings_bridged.parquet')
    save(holdings[['series_id', 'q', 'gvkey', 'value', 'report_date', 'filing_date']],
         'holdings_for_exposure.csv.gz', ['series_id', 'q', 'gvkey', 'value', 'filing_date'])
    labels = []
    for directory in sorted((source / 'firm_labels/vintages/02_label').iterdir()):
        if not directory.is_dir() or not directory.name.isdigit():
            continue
        frame = parquet('firm_labels/vintages/02_label/' + directory.name + '/firm_ai_labels.parquet')
        columns = ['gvkey', 'has_text', 'has_pat', 'has_evidence', 'ai_core']
        columns += [p + suffix for p in ('developer', 'enabler', 'adopter') for suffix in ('', '_1', '_2')]
        frame = frame[columns].copy()
        frame.insert(0, 'year', int(directory.name))
        labels.append(frame)
    if not labels:
        raise ValueError('No annual company labels found')
    save(pd.concat(labels, ignore_index=True), 'firm_label_vintages.csv', ['year', 'gvkey'])
    fields = ['accession', 'series_id', 'series_resolved', 'date', 'strategy_chars',
              'uses_ai', 'theme_broad', 'chain_developer', 'chain_enabler', 'chain_adopter']
    save(panel[fields], 'claim_history.csv', ['date', 'series_id', 'accession'])
    # Source hashes are data provenance, separate from the exported values.
    (destination / 'sources.json').write_text(json.dumps(dict(sorted(sources.items())), indent=2) + '\n',
                                              encoding='utf-8', newline='\n')
    print('Exported analysis inputs.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default=paths.OUTPUT,
                        help='intermediate data directory (default: THESIS_OUTPUT or data/interim)')
    parser.add_argument('--destination', default=str(paths.PROCESSED),
                        help='processed CSV directory (default: data/processed)')
    parser.add_argument('--headers', default=paths.data('485bpos_headers'),
                        help='SEC headers directory (default: THESIS_DATA/485bpos_headers)')
    args = parser.parse_args(argv)
    export(paths.project_path(args.source), paths.project_path(args.destination), paths.project_path(args.headers))


if __name__ == '__main__':
    main()

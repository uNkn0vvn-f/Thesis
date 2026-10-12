"""Shared loaders for the robustness and inference analyses.

The claim-holdings comparison is recomputed from the processed inputs so that measurement
variants (evidence-only labels, excluded firms, coverage thresholds) are built by one
function and checked against the pipeline's own exposure table. Tables go to
results/tables/robustness/ and figures to results/figures/.

Two label sets are available. THESIS_LABELS=luna (the default, the thesis measurement)
uses the fund and company labels of OpenAI's GPT-6 Luna built by luna_measures.py;
THESIS_LABELS=deepseek uses the original DeepSeek labels and writes its tables to
results/tables/robustness/deepseek/. Every script runs unchanged under either setting.
Directories come from src/paths.py, so THESIS_DATA and THESIS_OUTPUT apply here as in the
rest of the project.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src import paths as tp  # noqa: E402

# the project's directories (src/paths.py): THESIS_DATA and THESIS_OUTPUT redirect the raw
# and interim folders for every script that imports this module
RAW = Path(tp.DATA)
INTERIM = Path(tp.OUTPUT)
PROCESSED = Path(tp.PROCESSED)
RESULTS = Path(tp.RESULTS)
LABELS = os.environ.get('THESIS_LABELS', 'luna')
if LABELS not in ('luna', 'deepseek'):
    raise ValueError('THESIS_LABELS must be luna or deepseek')
TABLES_PRIMARY = RESULTS / 'tables' / 'robustness'
TABLES_DEEPSEEK = TABLES_PRIMARY / 'deepseek'
TABLES = TABLES_PRIMARY if LABELS == 'luna' else TABLES_DEEPSEEK
FIGURES = RESULTS / 'figures'
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

LUNA = INTERIM / 'cross_model' / 'gpt-6-luna'
LUNA_DERIVED = LUNA / 'derived'
PATHS = {
    'deepseek': {'claim_filings': PROCESSED / 'claim_filings.csv',
                 'labels': PROCESSED / 'firm_label_vintages.csv',
                 'panel': INTERIM / 'fund_claims' / 'panel_fund_filing.parquet',
                 'exposure': INTERIM / 'exposure' / 'fund_quarter_exposure.parquet',
                 'ladder': INTERIM / 'analysis' / 'flow_ladder' / 'out' / 'fund_month_ladder.parquet'},
    'luna': {'claim_filings': PROCESSED / 'claim_filings_luna.csv',
             'labels': PROCESSED / 'firm_label_vintages_luna.csv',
             'panel': LUNA_DERIVED / 'panel_fund_filing.parquet',
             'exposure': LUNA_DERIVED / 'fund_quarter_exposure.parquet',
             'ladder': INTERIM / 'analysis' / 'flow_ladder_luna' / 'out' / 'fund_month_ladder.parquet'},
}


def path(name: str, labels: str | None = None) -> Path:
    return PATHS[labels or LABELS][name]


# pre-event company labels and exposures of the event analysis: one information set per event
PREEVENT_DIRS = {'deepseek': INTERIM / 'analysis' / 'preevent',
            'luna': INTERIM / 'analysis' / 'preevent_luna'}


def preevent_path(tag: str, kind: str = 'exposure', labels: str | None = None) -> Path:
    d = PREEVENT_DIRS[labels or LABELS]
    if kind == 'exposure':
        return d / '03_exposure' / 'out' / tag / 'fund_quarter_exposure.parquet'
    return d / '02_label' / 'out' / tag / 'firm_ai_labels.parquet'

ROLES = ('developer', 'enabler', 'adopter')
SEED = 20261011

# Pre-specified before any variant was estimated (see README): the seven largest US
# technology-platform firms commonly called the "Magnificent Seven".
MAG7 = {'117768': 'NVIDIA', '012141': 'Microsoft', '001690': 'Apple', '064768': 'Amazon',
        '160329': 'Alphabet', '170617': 'Meta', '184996': 'Tesla'}


def load_exposure_table(labels: str | None = None) -> pd.DataFrame:
    """The fund-quarter exposure table, with benchmark-adjusted columns and as-of claims."""
    E = pd.read_parquet(path('exposure', labels))
    for c in ('uses_ai', 'theme_broad', 'chain_developer', 'chain_enabler', 'chain_adopter',
              'chain_unspecified', 'chain_benefit_vague'):
        E[c] = E[c].astype('boolean').fillna(False).astype(bool)
    E['series_id'] = E.series_id.astype(str)
    return E


def load_holdings() -> pd.DataFrame:
    H = pd.read_csv(PROCESSED / 'holdings_for_exposure.csv.gz',
                    dtype={'series_id': str, 'gvkey': str})
    H = H[H.value.notna() & (H.value > 0) & H.gvkey.notna()].copy()
    H['yr'] = H.q.str.slice(0, 4).astype(int)
    return H


def load_labels(labels: str | None = None) -> pd.DataFrame:
    L = pd.read_csv(path('labels', labels), dtype={'gvkey': str})
    return L.rename(columns={'year': 'yr'})


def load_claim_filings(labels: str | None = None) -> pd.DataFrame:
    """One row per classified fund filing (AI keyword in the strategy text)."""
    return pd.read_csv(path('claim_filings', labels), dtype={'series_id': str})


def load_claim_panel(labels: str | None = None, columns=None) -> pd.DataFrame:
    """Every fund filing with a strategy section; filings without AI text carry zeros."""
    return pd.read_parquet(path('panel', labels), columns=columns)


def load_buckets() -> pd.DataFrame:
    S = pd.read_parquet(INTERIM / 'holdings' / 'stock_quarter.parquet',
                        columns=['gvkey', 'q', 'bucket'])
    S['gvkey'] = S.gvkey.astype(str)
    return S


def build_exposure(H: pd.DataFrame, L: pd.DataFrame, *, evidence_only: bool = False,
                   drop: pd.DataFrame | None = None, buckets: pd.DataFrame | None = None
                   ) -> pd.DataFrame:
    """Fund-quarter role exposures on the labelled sleeve, as src.holdings.ai_exposure.

    evidence_only  a label with neither filing text nor a patent profile is treated as
                   unlabelled: it leaves numerator and coverage denominator together.
    drop           (gvkey, q) or (gvkey,) rows removed from the portfolio entirely before
                   shares and coverage are computed (exclusion of named firms).
    buckets        if given, also return size-decile x sector adjusted exposures.
    """
    X = H
    if drop is not None:
        keys = [c for c in ('gvkey', 'q') if c in drop.columns]
        X = X.merge(drop[keys].drop_duplicates().assign(_drop=True), on=keys, how='left')
        X = X[X._drop.isna()].drop(columns='_drop')
    X = X.merge(L[['gvkey', 'yr', 'has_evidence'] + list(ROLES)], on=['gvkey', 'yr'],
                how='left')
    X['seen'] = X.developer.notna()
    if evidence_only:
        X['seen'] &= X.has_evidence.astype('boolean').fillna(False).astype(bool)
    for r in ROLES:
        X['m_' + r] = np.where(X.seen, (X[r].fillna(-1) >= 2) * 100.0, np.nan)
    g = X.groupby(['series_id', 'q'], sort=False)
    base = g.agg(port_value=('value', 'sum')).reset_index()
    sub = X[X.seen]
    a = sub.groupby(['series_id', 'q']).agg(v_seen=('value', 'sum')).reset_index()
    for r in ROLES:
        w = (sub.value * sub['m_' + r]).groupby([sub.series_id, sub.q]).sum()
        a = a.merge(w.rename('_w' + r).reset_index(), on=['series_id', 'q'], how='left')
        a[r] = a['_w' + r] / a.v_seen
    base = base.merge(a[['series_id', 'q', 'v_seen'] + list(ROLES)], on=['series_id', 'q'],
                      how='left')
    base['labelled'] = 100 * base.v_seen.fillna(0) / base.port_value
    # value share of labelled holdings that carry no filing/patent evidence
    ne = (sub.value * (~sub.has_evidence.astype('boolean').fillna(False).astype(bool))
          ).groupby([sub.series_id, sub.q]).sum().rename('_ne').reset_index()
    base = base.merge(ne, on=['series_id', 'q'], how='left')
    base['noevidence_share'] = 100 * base._ne / base.v_seen
    base = base.drop(columns='_ne')
    if buckets is not None:
        Y = sub.merge(buckets, on=['gvkey', 'q'], how='left')
        F = Y[Y.bucket.notna()].drop_duplicates(['gvkey', 'q'])
        BM = (F.groupby(['q', 'bucket'])[['m_' + r for r in ROLES]].mean()
              .rename(columns={'m_' + r: 'e_' + r for r in ROLES}).reset_index())
        Y = Y.merge(BM, on=['q', 'bucket'], how='left')
        for r in ROLES:
            ok = Y['e_' + r].notna()
            t = (Y[ok].assign(_a=Y.value * Y['m_' + r], _e=Y.value * Y['e_' + r])
                 .groupby(['series_id', 'q']).agg(_a=('_a', 'sum'), _e=('_e', 'sum'),
                                                  _v=('value', 'sum')).reset_index())
            t['abn_' + r] = (t._a - t._e) / t._v
            base = base.merge(t[['series_id', 'q', 'abn_' + r]], on=['series_id', 'q'],
                              how='left')
    return base


def attach_claims(base: pd.DataFrame, E: pd.DataFrame) -> pd.DataFrame:
    """Contemporaneous prospectus claims from the pipeline's exposure table."""
    keep = ['series_id', 'q', 'report_date', 'filing_date', 'date', 'claim_age_days',
            'uses_ai', 'theme_broad', 'chain_developer', 'chain_enabler', 'chain_adopter',
            'chain_unspecified', 'chain_benefit_vague', 'tech', 'ai_rev']
    return base.merge(E[keep], on=['series_id', 'q'], how='inner', validate='one_to_one')

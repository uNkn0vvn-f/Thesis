"""Pre-event joint tests and non-AI technology checks of the event analysis.

Pre-event labels and exposures follow THESIS_LABELS (common.py).
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def source_path(path):
    """The source as recorded in the results: data/interim/... wherever THESIS_OUTPUT points."""
    from src import paths as tp
    return tp.portable(path)


from src.analysis.events import nonai_technology as I4  # noqa: E402
S2 = I4.S2
from src.analysis.robustness.common import INTERIM, LABELS, TABLES, preevent_path  # noqa: E402
I4.EX = {t: str(preevent_path(t)) for t in ('chatgpt', 'deepseek')}
I4.LAB = {t: str(preevent_path(t, 'label')) for t in ('chatgpt', 'deepseek')}
from src.analysis.events import single_event as S1  # noqa: E402
from src.analysis.flows.estimation import absorb, make_dims, absorbed_k, ols_cluster  # noqa: E402

DIAG = INTERIM / 'analysis' / 'events' / ('diagnostics_' + LABELS)
DIAG.mkdir(parents=True, exist_ok=True)
result = {'event_diagnostics': {}, 'nonai_technology': [], 'notes': [
    'Joint pre-event Wald tests use the full fund-cluster covariance matrix; sum of squared t statistics is not a joint test.',
    'The sector mapping is taken from the same current firm universe as the exposure builder.',
    'Placebo portfolios predate their placebo dates, but the cached company classifications are from the actual-event information sets; these are timing diagnostics, not fully reclassified placebo vintages.',
]}
L = S2.panel()
raw = pd.read_parquet(S2.LAD, columns=['series_id','month','nf_beg','na_beg_recon','ret'])
raw['month'] = pd.to_datetime(raw.month)
raw = raw[raw.na_beg_recon.ge(S2.MIN_TNA) & raw.nf_beg.notna() & np.isfinite(raw.nf_beg) & raw.ret.notna()].sort_values(['series_id','month'])
raw['month_number'] = raw.month.dt.year * 12 + raw.month.dt.month
raw['gap'] = raw.groupby('series_id').month_number.diff()
gaps = raw.loc[raw.gap.gt(1), ['series_id','month']].assign(noncalendar_lag=True)
result['lagged_flow_gap_observations_in_full_filtered_panel'] = len(gaps)

for tag, date in [('chatgpt', '2022-12-01'), ('deepseek', '2025-02-01')]:
    shock = pd.Timestamp(date)
    E = pd.read_parquet(I4.EX[tag])
    E['filing_date'] = pd.to_datetime(E.filing_date)
    E['report_date'] = pd.to_datetime(E.report_date)
    pre = E[(E.filing_date < I4.treat_cutoff(shock)) & E.labelled.ge(50)].sort_values('report_date').groupby('series_id').tail(1)
    stats = {'eligible_pre_event_funds': len(pre),
             'latest_filing': str(pre.filing_date.max().date()),
             'missing_claim_status': int(pre[['uses_ai','theme_broad']].isna().any(axis=1).sum()),
             'oldest_portfolio_days': int((I4.treat_cutoff(shock)-pre.report_date).dt.days.max()),
             'exposure_sha_source': source_path(I4.EX[tag])}
    for c in ['uses_ai','theme_broad']:
        E[c] = E[c].astype('boolean').fillna(False).astype(bool)
    D = S1.build(shock, E, L, set())
    joined = D.merge(gaps, on=['series_id','month'], how='left')
    stats['event_rows_with_noncalendar_lag'] = int(joined.noncalendar_lag.fillna(False).sum())
    ks = sorted(D.k.unique())
    cols = []
    for k in ks:
        if k != -1:
            col = 'x_' + str(k)
            D[col] = D.ai_pre * D.k.eq(k)
            cols.append(col)
    xn = cols + S1.XC
    sub, dims = make_dims(D, [['series_id'],['month']])
    Y = absorb(sub[['nf_w']].values, dims)[:,0]
    X = absorb(sub[xn].values, dims)
    b,se,V,G = ols_cluster(Y,X,sub.series_id.values,n_absorbed=absorbed_k(dims,sub.series_id.values))
    ix = [j for j,c in enumerate(cols) if int(c.split('_')[1]) < 0]
    v = V[np.ix_(ix,ix)]
    df = int(np.linalg.matrix_rank(v))
    wald = float(b[ix] @ np.linalg.pinv(v) @ b[ix])
    stats.update(pre_event_wald=wald, pre_event_df=df, pre_event_p=float(chi2.sf(wald,df)))
    result['event_diagnostics'][tag] = stats

# Use the identical SIC sector mapping and labelled sleeve as src.holdings.ai_exposure.
H = I4.holdings().drop(columns='sector')
U = pd.read_parquet(INTERIM / 'firm_labels' / 'firm_universe.parquet', columns=['gvkey','sector'])
U['gvkey'] = U.gvkey.astype(str)
H = H.merge(U, on='gvkey', how='left', validate='many_to_one')
for d1,d2,kind in I4.PAIRS:
    p1,diff1,_ = I4.exposure_file(H,'chatgpt',d1)
    p2,diff2,_ = I4.exposure_file(H,'deepseek',d2)
    assert max(diff1,diff2) < 1e-6, 'Technology mapping does not reproduce primary exposure'
    r,n,G = I4.stacked(L,p1,d1,p2,d2)
    result['nonai_technology'].append({'first_date': d1,'second_date': d2,'kind': kind,'n': n,'funds':G,
                                      **{k: {'coef': float(v[0]),'se': float(v[1]),'t': float(v[0]/v[1])} for k,v in r.items()}})
    print(kind, d1, d2, 'b=', r['b'], 'n=',n,flush=True)

(DIAG/'diagnostics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
(TABLES/'event_pretrends.json').write_text(json.dumps(dict(result, labels=LABELS),indent=2),encoding='utf-8')
rows=[]
for r in result['nonai_technology']:
    for term in ['a','b','chatgpt','deepseek']:
        rows.append({'first_date':r['first_date'],'second_date':r['second_date'],'kind':r['kind'],
                     'term':term,**r[term],'n':r['n'],'funds':r['funds']})
pd.DataFrame(rows).to_csv(TABLES/'event_nonai_technology.csv',index=False)
print(json.dumps(result['event_diagnostics'],indent=2),flush=True)

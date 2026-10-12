"""Check pre-period dynamics without treating the event month as pre-event evidence.

Pre-event exposures follow THESIS_LABELS (common.py)."""
from pathlib import Path
import os,sys,json
import numpy as np
import pandas as pd
from scipy.stats import chi2
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from src.analysis.events import nonai_technology as I4  # noqa: E402
S=I4.S2
from src.analysis.robustness.common import INTERIM, LABELS, TABLES, preevent_path  # noqa: E402
I4.EX={t: str(preevent_path(t)) for t in ('chatgpt','deepseek')}
# The monthly flow sample shared by both event windows.
L=S.panel()
out={}
strict_parts=[]
# Per event: a joint test of the pre-event coefficients relative to month -2, so the event
# month itself is not used as pre-event evidence.
for tag,date,sign in [('chatgpt','2022-12-01',1),('deepseek','2025-02-01',-1)]:
    shock=pd.Timestamp(date)
    D=S.window(tag,shock,sign,I4.EX[tag],L)
    D['k']=(D.month.dt.year-shock.year)*12+D.month.dt.month-shock.month
    E=pd.read_parquet(I4.EX[tag])
    pre=E[(E.filing_date<I4.treat_cutoff(shock)) & E.labelled.ge(50)].sort_values('report_date').groupby('series_id').tail(1)
    age=(I4.treat_cutoff(shock)-pre.set_index('series_id').report_date).dt.days
    selected=age.reindex(D.series_id.unique())
    d=D[D.k.le(-2)].copy()
    cols=[]
    for k in sorted(d.k.unique()):
        if k != -2:
            c='x_'+str(k);d[c]=d.ai_pre*d.k.eq(k);cols.append(c)
    sub,dims=S.make_dims(d,[['series_id'],['month']])
    y=S.absorb(sub[['nf_w']].values,dims)[:,0]
    x=S.absorb(sub[cols+S.XC].values,dims)
    b,se,V,G=S.ols_cluster(y,x,sub.series_id.values,n_absorbed=S.absorbed_k(dims,sub.series_id.values))
    ix=list(range(len(cols)));v=V[np.ix_(ix,ix)]
    df=int(np.linalg.matrix_rank(v));w=float(b[ix]@np.linalg.pinv(v)@b[ix])
    out[tag]={'reference_k':-2,'event_month_excluded':True,'pre_only_wald':w,'df':df,'p':float(chi2.sf(w,df)),
              'estimation_funds_with_portfolio_older_than_366_days':int(selected.gt(366).sum()),
              'max_estimation_portfolio_age_days':int(selected.max())}
    # Keep fund-months whose lagged flow is the calendar month before and whose pre-event
    # portfolio is at most 366 days old.
    raw=pd.read_parquet(S.LAD,columns=['series_id','month','nf_beg','na_beg_recon','ret'])
    raw['month']=pd.to_datetime(raw.month)
    raw=raw[raw.na_beg_recon.ge(S.MIN_TNA)&np.isfinite(raw.nf_beg)&raw.ret.notna()].sort_values(['series_id','month'])
    raw['m']=raw.month.dt.year*12+raw.month.dt.month
    raw['g']=raw.groupby('series_id').m.diff()
    adjacent=set(zip(raw.loc[raw.g.eq(1),'series_id'],raw.loc[raw.g.eq(1),'month']))
    mask=[(s,m) in adjacent for s,m in zip(D.series_id,D.month)]
    fresh=D.series_id.map(age).le(366)
    strict_parts.append(D.loc[np.asarray(mask)&fresh].copy())
# The stacked design re-estimated on that restricted sample.
DD=pd.concat(strict_parts,ignore_index=True)
for a,c in [('ai_post','ai_pre'),('cl_post','claim_pre')]:
    DD[a]=DD[c]*DD.post;DD[a+'_sign']=DD[a]*DD.sign
terms=['ai_post','ai_post_sign','cl_post','cl_post_sign']+S.XC
b,se,V,n,G=S.fit(DD,terms,None)
out['timing_sensitivity']={'restriction':'Calendar-adjacent lagged flow and selected pre-event holdings no older than 366 days',
                           'n':n,'funds':G,'direction_coef':float(b[1]),'direction_se':float(se[1])}
# Write the diagnostics to the interim folder and the results table.
p=INTERIM/'analysis'/'events'/('diagnostics_'+LABELS)/'timing_diagnostics.json'
p.parent.mkdir(parents=True,exist_ok=True)
p.write_text(json.dumps(out,indent=2),encoding='utf-8')
(TABLES/'event_timing.json').write_text(json.dumps(dict(out,labels=LABELS),indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))

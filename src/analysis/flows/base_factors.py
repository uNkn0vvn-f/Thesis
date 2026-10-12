"""Monthly base factors: Carhart's four and two industry portfolios.

Writes analysis/factors/out/factors_monthly.parquet, the factor input of the AI factor
(src.analysis.flows.ai_factor) and of the alphas of the alignment comparison. Columns, all
in percent per month:

    mktrf smb hml rf umd   the Fama-French and momentum factors, daily values compounded
                           within the month (exact for the market and the risk-free rate,
                           the usual approximation for the long-short factors)
    r_all                  value-weighted return of US common stocks on NYSE, AMEX and
                           Nasdaq (CRSP monthly stock file, CIZ format), weighted by the
                           previous month's market capitalisation
    r_enab, enab           the same for the enabler industries (semiconductors, computer
                           and communications hardware, electronic components, computer
                           processing services) and that return net of the market
    r_tech, tech           the same for a broader information-technology set

The industry portfolios stop where the CRSP monthly file stops; the Fama-French factors
reach further, and the later months carry only those.

Inputs: THESIS_DATA/wrds/ff_daily.parquet and THESIS_DATA/wrds/crsp_msf_v2.parquet
(src.data.pull_wrds).

    python -m src.analysis.flows.base_factors
"""
import io
import os

import numpy as np
import pandas as pd

from src import paths as tp

W = tp.data('wrds')
OUT = tp.out('analysis', 'factors', 'out')
LOGS = tp.out('analysis', 'factors', 'logs')

ENABLER_CORE_SIC = {3674, 3672, 3559, 3827, 3571, 3572, 3575, 3576, 3577,
                    3661, 3663, 3669, 3678, 3679, 7374}
TECH_SIC = ENABLER_CORE_SIC | {3570, 3573, 3574, 3578, 3579, 3651, 3662, 3670,
                               3671, 3675, 3676, 3677, 3680, 3690, 3812, 3825,
                               3826, 7370, 7371, 7372, 7373, 7375, 7376, 7377,
                               7378, 7379}
FF = ['mktrf', 'smb', 'hml', 'rf', 'umd']


def main():
    """Write the monthly factor file from the daily Fama-French factors and the CRSP monthly
    file."""
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    buf = io.StringIO()

    def w(s=''):
        buf.write(s + '\n')
        print(s)

    F = pd.read_parquet(os.path.join(W, 'ff_daily.parquet'))
    F['date'] = pd.to_datetime(F.date)
    F['month'] = F.date.dt.to_period('M').dt.to_timestamp()
    for c in FF:
        F[c] = pd.to_numeric(F[c], errors='coerce')
    M = F.groupby('month')[FF].apply(lambda g: (1 + g).prod() - 1).reset_index()
    M[FF] *= 100
    w('Fama-French / Carhart: %d months, %s .. %s, daily factors compounded within the month'
      % (len(M), M.month.min().date(), M.month.max().date()))

    # US common stocks on NYSE, AMEX and Nasdaq, weighted by the previous month's market
    # cap.
    S = pd.read_parquet(os.path.join(W, 'crsp_msf_v2.parquet'),
                        columns=['permno', 'mthcaldt', 'mthret', 'mthcap', 'siccd', 'sharetype',
                                 'securitytype', 'securitysubtype', 'issuertype', 'primaryexch'])
    S['month'] = pd.to_datetime(S.mthcaldt).dt.to_period('M').dt.to_timestamp()
    S = S[(S.sharetype == 'NS') & (S.securitytype == 'EQTY') & (S.securitysubtype == 'COM')
          & S.issuertype.isin(['CORP', 'ACOR']) & S.primaryexch.isin(['N', 'A', 'Q'])]
    S['mthret'] = pd.to_numeric(S.mthret, errors='coerce')
    S['siccd'] = pd.to_numeric(S.siccd, errors='coerce')
    S = S[S.mthret.notna() & S.mthcap.notna() & (S.mthcap > 0)].sort_values(['permno', 'month'])
    S['me_lag'] = S.groupby('permno').mthcap.shift()
    S = S[S.me_lag.notna() & (S.me_lag > 0)]

    def vw(sub):
        """Value-weighted return of `sub`, in percent."""
        return 100 * np.average(sub.mthret, weights=sub.me_lag)

    # Value-weighted returns of all stocks and of the enabler and technology SIC sets.
    cols = {'r_all': S.groupby('month').apply(vw, include_groups=False)}
    counts = {}
    for nm, sic in (('enab', ENABLER_CORE_SIC), ('tech', TECH_SIC)):
        sub = S[S.siccd.isin(sic)]
        cols['r_' + nm] = sub.groupby('month').apply(vw, include_groups=False)
        counts[nm] = sub.groupby('month').permno.nunique()
    P = pd.DataFrame(cols).reset_index()
    M = M.merge(P, on='month', how='left')
    M['enab'] = M.r_enab - M.r_all
    M['tech'] = M.r_tech - M.r_all
    w('industry portfolios: stocks per month, enabler median %d, technology median %d'
      % (counts['enab'].median(), counts['tech'].median()))
    w('%-8s %9s %9s %9s %9s' % ('factor', 'mean', 'sd', 'min', 'max'))
    for c in ('mktrf', 'smb', 'hml', 'umd', 'enab', 'tech'):
        w('%-8s %9.3f %9.3f %9.3f %9.3f' % (c, M[c].mean(), M[c].std(), M[c].min(), M[c].max()))
    C = M[['mktrf', 'smb', 'hml', 'umd', 'enab', 'tech']].corr()
    w('enab and tech correlate %.2f; the alpha models use one at a time' % C.loc['enab', 'tech'])
    w('CRSP monthly reaches %s, the Fama-French factors %s; enab and tech are missing for %d months'
      % (M.loc[M.enab.notna(), 'month'].max().date(), M.month.max().date(), int(M.enab.isna().sum())))
    M.to_parquet(os.path.join(OUT, 'factors_monthly.parquet'), index=False)
    w('wrote %s (%d months)' % (os.path.join(OUT, 'factors_monthly.parquet'), len(M)))
    with io.open(os.path.join(LOGS, 'base_factors.txt'), 'w', encoding='utf-8') as fh:
        fh.write(buf.getvalue())


if __name__ == '__main__':
    main()

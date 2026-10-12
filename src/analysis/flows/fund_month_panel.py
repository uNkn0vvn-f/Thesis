"""Fund-month flows and returns from the SEC N-PORT data sets, with N-CEN fund types.

Two monthly panels for the flow analyses, built from the quarterly N-PORT data sets that
src.data.download_nport fetches (every NPORT-P filing covers one fund series and reports
three months):

    analysis/fund_flows/out/fund_month_flows.parquet
        one row per fund series and month: gross sales, reinvestments and redemptions
        (FUND_REPORTED_INFO, items B.6), the quarter-end net assets, the fund's monthly
        return, the net flow, the beginning-of-month net assets reconstructed from the
        accounting identity, and the flow rates nf_qend (over quarter-end assets) and
        nf_beg (over reconstructed beginning-of-month assets), both in percent
    analysis/monthly_returns/out/fund_month_returns.parquet
        one row per fund series and month: the median monthly total return across the
        fund's share classes (MONTHLY_TOTAL_RETURN, item B.5), the equal-weighted mean,
        the class range, and the fund-type flags of N-CEN (ETF, index, money market,
        target date, fund of funds, interval, master-feeder)

Three choices are decided by the data rather than assumed, and the run log records each:

    MONTH ORDER   the data sets do not say which calendar month MON1 is. Both conventions
                  (MON1 the oldest of the three months; MON1 the report month) are scored
                  against the CRSP value-weighted market return; the one that correlates
                  is used (MON1 is the oldest).
    FLOW SIGN     REDEMPTION_FLOW is not consistently signed by filers (negative in about a
                  third of pre-2024 filings). Three definitions, sales - redemptions,
                  sales - |redemptions| and sales + redemptions, are tested against the
                  identity NA_t = NA_(t-1) (1 + r_t) + F_t rolled through each quarter;
                  the one with the smallest residual is used (sales - |redemptions|).
    DENOMINATOR   net assets are reported at the quarter end only. Beginning-of-month
                  assets are reconstructed backwards from that figure with each month's
                  return and flow; the quarter-end scaling is kept beside it.

Share classes are collapsed to the fund with the median class return, the convention of
the flow panel; N-PORT carries no class-level assets, so no investor-weighted return is
built. Fund-type flags are blank when false in N-CEN, so a blank is read as "no".

Inputs: THESIS_DATA/nport/*_nport.zip and THESIS_DATA/ncen/*_ncen.zip
(src.data.download_nport) and THESIS_DATA/wrds/crsp_msi.parquet (src.data.pull_wrds).

    python -m src.analysis.flows.fund_month_panel
"""
import csv
import glob
import io
import os
import re
import zipfile

import numpy as np
import pandas as pd

from src import paths as tp

NPORT = tp.data('nport')
NCEN = tp.data('ncen')
MSI = tp.data('wrds', 'crsp_msi.parquet')
OUT_FLOWS = tp.out('analysis', 'fund_flows', 'out')
LOGS_FLOWS = tp.out('analysis', 'fund_flows', 'logs')
OUT_RETURNS = tp.out('analysis', 'monthly_returns', 'out')
LOGS_RETURNS = tp.out('analysis', 'monthly_returns', 'logs')

FLOW_FIELDS = ['NET_ASSETS'] + ['%s_FLOW_MON%d' % (a, k) for k in (1, 2, 3)
                                for a in ('SALES', 'REINVESTMENT', 'REDEMPTION')]
FLAGS = ['IS_ETF', 'IS_INDEX', 'IS_MONEY_MARKET', 'IS_TARGET_DATE', 'IS_FUND_OF_FUND',
         'IS_INTERVAL', 'IS_MASTER_FEEDER']
FIRST_MONTH, LAST_MONTH = '2019-01-01', '2026-06-30'
csv.field_size_limit(1 << 24)


# A run log that prints each line and keeps it for the log file.
class Log:
    def __init__(self):
        """Start with an empty log."""
        self.buf = io.StringIO()

    def __call__(self, s=''):
        """Print a line and keep it."""
        self.buf.write(s + '\n')
        try:
            print(s)
        except UnicodeEncodeError:
            print(s.encode('ascii', 'replace').decode())

    def save(self, folder, name):
        """Write the kept lines to folder/name."""
        os.makedirs(folder, exist_ok=True)
        with io.open(os.path.join(folder, name), 'w', encoding='utf-8') as fh:
            fh.write(self.buf.getvalue())


def tsv(z, name):
    """The column index and a row reader of one TSV inside a data-set ZIP; empty if absent."""
    if name not in z.namelist():
        return None, iter(())
    t = io.TextIOWrapper(z.open(name), encoding='utf-8', errors='replace', newline='')
    rd = csv.reader(t, delimiter='\t')
    try:
        header = next(rd)
    except StopIteration:
        return None, iter(())
    return {c: j for j, c in enumerate(header)}, rd


def read_nport():
    """Fund-level flows and class-level returns of every quarterly data set, as filed.

    The report date comes from the submission table; the series of a return row from
    the fund row of the same accession (one NPORT-P covers one series)."""
    flows, rets = [], []
    for p in sorted(glob.glob(os.path.join(NPORT, '*_nport.zip'))):
        q = re.search(r'(\d{4}q\d)', os.path.basename(p)).group(1)
        z = zipfile.ZipFile(p)
        i, rd = tsv(z, 'SUBMISSION.tsv')
        report = {}
        if i:
            dc = next((c for c in ('REPORT_DATE', 'PERIOD_OF_REPORT', 'REPORT_ENDING_PERIOD')
                       if c in i), None)
            if dc:
                report = {r[i['ACCESSION_NUMBER']]: r[i[dc]] for r in rd}
        i, rd = tsv(z, 'FUND_REPORTED_INFO.tsv')
        if not i or any(c not in i for c in FLOW_FIELDS + ['SERIES_ID', 'ACCESSION_NUMBER']):
            continue
        series = {}
        for r in rd:
            acc = r[i['ACCESSION_NUMBER']]
            series.setdefault(acc, r[i['SERIES_ID']])
            flows.append([q, report.get(acc, ''), r[i['SERIES_ID']]] + [r[i[c]] for c in FLOW_FIELDS])
        i, rd = tsv(z, 'MONTHLY_TOTAL_RETURN.tsv')
        if i:
            for r in rd:
                acc = r[i['ACCESSION_NUMBER']]
                rets.append([series.get(acc, ''), r[i['CLASS_ID']], report.get(acc, '')]
                            + [r[i['MONTHLY_TOTAL_RETURN%d' % k]] for k in (1, 2, 3)])
    F = pd.DataFrame(flows, columns=['q', 'period', 'series_id'] + [c.lower() for c in FLOW_FIELDS])
    for c in [c.lower() for c in FLOW_FIELDS]:
        F[c] = pd.to_numeric(F[c], errors='coerce')
    R = pd.DataFrame(rets, columns=['series_id', 'class_id', 'report_date', 'm1', 'm2', 'm3'])
    R = R[(R.series_id != '') & (R.report_date != '')]
    for c in ('m1', 'm2', 'm3'):
        R[c] = pd.to_numeric(R[c], errors='coerce')
    return F, R


def class_returns(R, log):
    """Class-month returns, with the month convention scored against the CRSP market."""
    R = R.copy()
    R['report_date'] = pd.to_datetime(R.report_date, errors='coerce', format='mixed')
    R = R[R.report_date.notna()]
    log('  class-filing rows %d, %d series, %d classes, %s .. %s'
        % (len(R), R.series_id.nunique(), R.class_id.nunique(),
           R.report_date.min().date(), R.report_date.max().date()))
    long = R.melt(id_vars=['series_id', 'class_id', 'report_date'],
                  value_vars=['m1', 'm2', 'm3'], var_name='slot', value_name='ret')
    long = long[long.ret.notna()].copy()
    long['k'] = long.slot.str[1].astype(int)
    long['month_A'] = (long.report_date.dt.to_period('M') - (3 - long.k)).dt.to_timestamp()
    long['month_B'] = (long.report_date.dt.to_period('M') - (long.k - 1)).dt.to_timestamp()
    log()
    log('-' * 96)
    log('1. WHICH CALENDAR MONTH IS MON1?   scored against CRSP, not assumed')
    log('-' * 96)
    best = 'month_A'
    if os.path.exists(MSI):
        msi = pd.read_parquet(MSI)
        msi['month'] = pd.to_datetime(msi.date).dt.to_period('M').dt.to_timestamp()
        mkt = msi[['month', 'vwretd']].dropna()
        bestr = -9
        for cand, desc in (('month_A', 'MON1 = oldest of the three'),
                           ('month_B', 'MON1 = the report month')):
            agg = (long.groupby(cand).ret.median().rename('fund').reset_index()
                   .rename(columns={cand: 'month'}))
            j = agg.merge(mkt, on='month', how='inner')
            j = j[(j.month >= '2020-01-01') & (j.month <= '2026-06-30')]
            r = np.corrcoef(j.fund, 100 * j.vwretd)[0, 1]
            log('  %-28s n = %3d months   corr %+.3f' % (desc, len(j), r))
            if r > bestr:
                best, bestr = cand, r
        log('  -> using %s (corr %.3f)' % (best, bestr))
        if bestr < 0.9:
            log('  WARNING: neither convention fits; do not use these panels as dated')
    else:
        log('  %s not found; MON1 taken as the oldest month without the CRSP check' % MSI)
    long['month'] = long[best]
    long = long[(long.month >= FIRST_MONTH) & (long.month <= LAST_MONTH)]
    return long.sort_values('report_date').drop_duplicates(['class_id', 'month'], keep='last')


def fund_returns(cm, log):
    """One monthly return per fund: the median across its share classes, with the equal-
    weighted mean and the class range alongside."""
    g = cm.groupby(['series_id', 'month']).ret
    fm = pd.DataFrame({'ret': g.median(), 'ret_ew': g.mean(), 'ret_min': g.min(),
                       'ret_max': g.max(), 'n_class': g.size()}).reset_index()
    fm['class_spread'] = fm.ret_max - fm.ret_min
    fm = fm[fm.ret.between(-90, 200)]
    log()
    log('-' * 96)
    log('2. COLLAPSING SHARE CLASSES  (the spread across classes is mostly the fee spread)')
    log('-' * 96)
    multi = fm[fm.n_class > 1]
    log('  fund-months with >1 class: %d of %d (%.1f%%), median %d classes'
        % (len(multi), len(fm), 100 * len(multi) / len(fm), multi.n_class.median()))
    log('  spread (max - min class return) within a fund-month, pp: median %.3f  p90 %.3f  p99 %.3f'
        % (multi.class_spread.median(), multi.class_spread.quantile(.9),
           multi.class_spread.quantile(.99)))
    log('  median vs equal-weighted mean: mean absolute difference %.4f pp'
        % (fm.ret - fm.ret_ew).abs().mean())
    log('  `ret` is the median class return; `ret_ew` the equal-weighted mean. N-PORT has')
    log('  no class-level net assets, so no investor-weighted fund return is built.')
    return fm


def ncen_attributes(log):
    """Fund-type flags from N-CEN, any filing of the series; a blank flag means no."""
    recs = []
    for p in sorted(glob.glob(os.path.join(NCEN, '*_ncen.zip'))):
        try:
            z = zipfile.ZipFile(p)
            D = pd.read_csv(io.BytesIO(z.read('FUND_REPORTED_INFO.tsv')), sep='\t',
                            dtype=str, low_memory=False)
            S = pd.read_csv(io.BytesIO(z.read('SUBMISSION.tsv')), sep='\t',
                            dtype=str, low_memory=False)
        except (KeyError, zipfile.BadZipFile):
            continue
        dcol = next((c for c in ('PERIOD_ENDING', 'REPORT_ENDING_PERIOD', 'FILING_DATE')
                     if c in S.columns), None)
        S = S[['ACCESSION_NUMBER'] + ([dcol] if dcol else [])]
        D = D.merge(S, on='ACCESSION_NUMBER', how='left')
        keep = (['SERIES_ID'] + [c for c in FLAGS if c in D.columns]
                + [c for c in ('RETURN_B4_FEES_AND_EXPENSES', 'RETURN_AFTR_FEES_AND_EXPENSES')
                   if c in D.columns] + ([dcol] if dcol else []))
        recs.append(D[keep].rename(columns={dcol: 'period'} if dcol else {}))
    N = pd.concat(recs, ignore_index=True)
    N = N[N.SERIES_ID.notna()]
    N['period'] = pd.to_datetime(N.get('period'), errors='coerce', format='mixed')
    for c in FLAGS:
        if c in N.columns:
            N[c] = N[c].fillna('N').str.upper().eq('Y')
    log()
    log('-' * 96)
    log('3. WHAT KIND OF FUND  (N-CEN; a blank flag means no, never missing)')
    log('-' * 96)
    log('  N-CEN fund-year rows %d over %d series' % (len(N), N.SERIES_ID.nunique()))
    ever = N.groupby('SERIES_ID')[[c for c in FLAGS if c in N.columns]].any()
    for c in ever.columns:
        log('  %-22s %10d series %6.1f%%' % (c, int(ever[c].sum()), 100 * ever[c].mean()))
    log('  The N-CEN return fields are item C.3.j index-tracking disclosure, filled only')
    log('  for index funds; they are carried for that use and are not a fee measure.')
    attr = ever.reset_index().rename(columns={'SERIES_ID': 'series_id'})
    attr.columns = [c.lower() for c in attr.columns]
    return N, attr


def returns_panel(R, log):
    """Write the monthly return panel with the N-CEN fund-type flags, and return it."""
    cm = class_returns(R, log)
    fm = fund_returns(cm, log)
    N, attr = ncen_attributes(log)
    fm = fm.merge(attr, on='series_id', how='left')
    for c in [c.lower() for c in FLAGS]:
        if c in fm.columns:
            fm[c] = fm[c].astype('boolean').fillna(False).astype(bool)
    fm['ncen_known'] = fm.series_id.isin(set(attr.series_id))
    os.makedirs(OUT_RETURNS, exist_ok=True)
    fm.to_parquet(os.path.join(OUT_RETURNS, 'fund_month_returns.parquet'), index=False)
    N.to_parquet(os.path.join(OUT_RETURNS, 'ncen_fund_attributes.parquet'), index=False)
    log()
    log('  return panel %d fund-months, %d funds, %s .. %s'
        % (len(fm), fm.series_id.nunique(), fm.month.min().date(), fm.month.max().date()))
    log('wrote %s' % os.path.join(OUT_RETURNS, 'fund_month_returns.parquet'))
    return fm


def flows_panel(N, R, log):
    """Build, check and write the monthly flow panel from the N-PORT flow fields and the
    fund returns."""
    N = N.copy()
    N['period'] = pd.to_datetime(N.period, errors='coerce', format='mixed')
    N = N[N.period.notna() & N.series_id.notna()]
    log('  filings %d over %d series, %s .. %s'
        % (len(N), N.series_id.nunique(), N.period.min().date(), N.period.max().date()))
    log()
    log('-' * 96)
    log('1. THE SIGN OF REDEMPTION_FLOW IS NOT CONSISTENT')
    log('-' * 96)
    for k in (1, 2, 3):
        v = N['redemption_flow_mon%d' % k]
        log('  MON%d  negative %5.1f%%   positive %5.1f%%   zero %5.1f%%'
            % (k, 100 * (v < 0).mean(), 100 * (v > 0).mean(), 100 * (v == 0).mean()))
    log('  "sales - redemption" therefore adds redemptions wherever the filer signed them')
    log('  negative; the candidates below are tested against the net-asset identity.')
    log()

    # One row per fund and month: each filing reports three months, MON1 the oldest.
    recs = []
    for k in (1, 2, 3):
        off = 3 - k                       # MON1 is the oldest of the three months
        recs.append(pd.DataFrame({
            'series_id': N.series_id, 'q': N.q, 'period': N.period,
            'month': (N.period.dt.to_period('M') - off).dt.to_timestamp(),
            'sales': N['sales_flow_mon%d' % k], 'reinv': N['reinvestment_flow_mon%d' % k],
            'redeem': N['redemption_flow_mon%d' % k], 'na_qend': N.net_assets, 'k': k}))
    F = pd.concat(recs, ignore_index=True).dropna(subset=['month'])
    F = F.sort_values(['series_id', 'month', 'period']).drop_duplicates(['series_id', 'month'], keep='last')
    # Three candidate flow definitions, scored against the net-asset identity below.
    F['flow_A'] = F.sales - F.redeem
    F['flow_B'] = F.sales - F.redeem.abs()
    F['flow_C'] = F.sales + F.redeem
    F = F.merge(R[['series_id', 'month', 'ret']], on=['series_id', 'month'], how='left')

    log('-' * 96)
    log('2. WHICH FLOW DEFINITION SATISFIES  NA_t = NA_(t-1)(1 + r_t) + F_t ?')
    log('-' * 96)
    log('  The previous quarter-end net assets are rolled forward through the three months')
    log('  of a filing with each month\'s own return and flow and compared with the observed')
    log('  quarter-end figure (returns are in percent).')
    # Roll the previous quarter-end net assets forward through the quarter under each
    # definition and compare with the observed quarter end.
    F['r'] = F.ret / 100.0
    piv = F.pivot_table(index=['series_id', 'period'], columns='k',
                        values=['r', 'flow_A', 'flow_B', 'flow_C'], aggfunc='first')
    piv.columns = ['%s_%d' % (a, b) for a, b in piv.columns]
    piv = piv.reset_index()
    na = F[F.k == 3][['series_id', 'period', 'na_qend']].drop_duplicates(['series_id', 'period'])
    piv = piv.merge(na, on=['series_id', 'period'], how='left').sort_values(['series_id', 'period'])
    piv['na_prev'] = piv.groupby('series_id').na_qend.shift()
    piv['gap'] = (piv.period - piv.groupby('series_id').period.shift()).dt.days
    T = piv[piv.gap.between(80, 100) & piv.na_prev.notna() & (piv.na_prev > 0) & (piv.na_qend > 0)].copy()
    res = {}
    for cand in ('A', 'B', 'C'):
        need = ['r_1', 'r_2', 'r_3'] + ['flow_%s_%d' % (cand, k) for k in (1, 2, 3)]
        g = T.dropna(subset=need)
        v = g.na_prev.astype(float).values
        for k in (1, 2, 3):
            v = v * (1 + g['r_%d' % k].values) + g['flow_%s_%d' % (cand, k)].values
        err = np.abs(v - g.na_qend.values) / g.na_qend.values
        res[cand] = (np.median(err), (err < 0.01).mean(), (err < 0.05).mean(), len(err))
    label = {'A': 'A  sales - redemptions', 'B': 'B  sales - |redemptions|', 'C': 'C  sales + redemptions'}
    log('  %-28s %14s %11s %11s %9s' % ('candidate', 'median |resid|', 'within 1%', 'within 5%', 'n'))
    for c in ('A', 'B', 'C'):
        m, w1, w5, n = res[c]
        log('  %-28s %13.4f %10.1f%% %10.1f%% %9d' % (label[c], m, 100 * w1, 100 * w5, n))
    best = min(res, key=lambda c: res[c][0])
    log('  -> using candidate %s' % label[best])
    log('  The residual is not zero under any definition: net assets are reported for the')
    log('  series while returns are class medians, and classes launched inside a quarter')
    log('  break the identity. What matters is which candidate is closest.')
    log()
    F['flow'] = F['flow_' + best]

    F = F.sort_values(['series_id', 'period', 'k'])
    # roll the observed quarter-end value backwards through the quarter:
    #   na_beg(m) = (na_end(m) - F_m) / (1 + r_m),  na_end(m) = na_beg(m + 1)
    F['na_end_m'] = np.nan
    F['na_beg_recon'] = np.nan
    piv2 = F.pivot_table(index=['series_id', 'period'], columns='k', values=['r', 'flow'], aggfunc='first')
    piv2.columns = ['%s_%d' % (a, b) for a, b in piv2.columns]
    piv2 = piv2.reset_index().merge(na, on=['series_id', 'period'], how='left')
    end3 = piv2.na_qend.astype(float)
    beg3 = (end3 - piv2.flow_3) / (1 + piv2.r_3)
    beg2 = (beg3 - piv2.flow_2) / (1 + piv2.r_2)
    beg1 = (beg2 - piv2.flow_1) / (1 + piv2.r_1)
    B = pd.concat([pd.DataFrame({'series_id': piv2.series_id, 'period': piv2.period, 'k': k, 'na_beg_recon': v})
                   for k, v in ((1, beg1), (2, beg2), (3, beg3))], ignore_index=True)
    F = F.drop(columns=['na_beg_recon']).merge(B, on=['series_id', 'period', 'k'], how='left')
    F['na_beg_recon'] = F.na_beg_recon.where(F.na_beg_recon > 0)
    # Flow rates in percent, over quarter-end and over reconstructed beginning-of-month
    # assets.
    F['nf_qend'] = 100 * F.flow / F.na_qend.replace(0, np.nan)
    F['nf_beg'] = 100 * F.flow / F.na_beg_recon.replace(0, np.nan)

    log('-' * 96)
    log('3. THE DENOMINATOR')
    log('-' * 96)
    for nm, c in (('quarter-end net assets', 'nf_qend'), ('reconstructed beginning-of-month assets', 'nf_beg')):
        v = F[c].replace([np.inf, -np.inf], np.nan).dropna()
        log('  %-42s median %6.2f%%  p1 %7.2f  p99 %7.2f  |>50%%| %5.2f%%'
            % (nm, v.median(), v.quantile(.01), v.quantile(.99), 100 * (v.abs() > 50).mean()))
    log('  Both are carried; the analyses use nf_beg, a month of flow over the assets at')
    log('  the start of that month.')
    log()
    os.makedirs(OUT_FLOWS, exist_ok=True)
    F.to_parquet(os.path.join(OUT_FLOWS, 'fund_month_flows.parquet'), index=False)
    log('  flow panel %d fund-months over %d series, %s .. %s'
        % (len(F), F.series_id.nunique(), F.month.min().date(), F.month.max().date()))
    log('wrote %s' % os.path.join(OUT_FLOWS, 'fund_month_flows.parquet'))
    return F


def main():
    """Build the return panel, then the flow panel on its returns."""
    N, R = read_nport()
    log = Log()
    log('=' * 96)
    log('FUND-MONTH RETURN PANEL')
    log('=' * 96)
    fm = returns_panel(R, log)
    log.save(LOGS_RETURNS, 'fund_month_panel.txt')
    log = Log()
    log('=' * 96)
    log('FUND-MONTH FLOW PANEL')
    log('=' * 96)
    flows_panel(N, fm[['series_id', 'month', 'ret']], log)
    log.save(LOGS_FLOWS, 'fund_month_panel.txt')


if __name__ == '__main__':
    main()

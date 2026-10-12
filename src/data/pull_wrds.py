"""Pull the WRDS tables the pipeline reads, one parquet file each, into THESIS_DATA/wrds.

    comp_company           CIK of each Compustat company, to find its 10-K and 20-F filings
    comp_security          North America security identifiers, to bridge holdings to gvkey
    comp_g_security        Compustat Global security identifiers (ISIN), the same for
                           non-US holdings
    comp_secm              monthly price and shares outstanding, for size benchmarks
    comp_funda             annual fundamentals: revenue and sector codes
    ccm_linktable          CRSP-Compustat link
    crsp_msi               monthly market return, to date the N-PORT return months
    crsp_msf_v2            monthly security file (CIZ format): returns, market caps and SIC
                           codes for the base factors and the AI factor
    crsp_dsf_v2_2020_2025  daily US common-stock returns and market caps (shift-share design)
    ff_daily               daily Fama-French factors (month alignment check of the flow panel)
    patents_permno_cpc     USPTO patents linked to firms, with their primary CPC subclass

The account needs the Compustat, CRSP, Fama-French and patent-firm-link subscriptions.
Set TABLES to a comma-separated subset; a table already on disk is not pulled again.
"""
import os, sys, time

from src import paths as tp  # noqa: E402

DEST = os.environ.get('DEST', tp.data('wrds'))
BEG = os.environ.get('BEG', '2019-01-01')
END = os.environ.get('END', '2026-12-31')
ONLY = {t.strip() for t in os.environ.get('TABLES', '').split(',') if t.strip()}

QUERIES = {
    # CIK of every Compustat company, used to locate its annual reports on EDGAR
    'comp_company': ("""
        select gvkey, cik, conm, fic, sic, naics from comp.company
    """, []),

    # daily Fama-French factors from 2020, for the month alignment check of the flow panel
    'ff_daily': ("""
        select date, mktrf, smb, hml, rf, umd
        from ff.factors_daily
        where date >= '2020-01-01'
    """, []),

    # USPTO patents linked to CRSP/Compustat firms, with the examiner's primary CPC class
    'patents_permno_cpc': ("""
        select permno, gvkey, patent_num, filing_year, grant_year, primary_cpc_class
        from contrib_patent_firm_link.patent_level
        where permno is not null and filing_year >= '1990'
    """, []),

    # monthly security file in the CIZ format, from mid-2019, every security type (the
    # factor modules select US common stocks); the columns of the stored file
    'crsp_msf_v2': ("""
        select permno, mthcaldt, mthprc, mthcap, mthret, shrout, cusip, ticker, usincflg,
               issuertype, securitytype, securitysubtype, sharetype, primaryexch, siccd
        from crsp.msf_v2
        where mthcaldt >= '2019-06-01'
    """, []),

    # daily stock file (US common shares) for the event-day returns and market caps that the
    # event design's shift-share inference uses (src/analysis/events, run by
    # src/analysis/robustness/event_shift_share.py).
    # About 6.3 million rows, 2020 onwards; TABLES=crsp_dsf_v2_2020_2025 pulls it alone.
    'crsp_dsf_v2_2020_2025': ("""
        select permno, dlycaldt, dlyret, dlyprc, dlycap
        from crsp.dsf_v2
        where dlycaldt >= '2020-01-01'
          and sharetype='NS' and securitytype='EQTY' and securitysubtype='COM'
          and usincflg='Y'
    """, []),

    # monthly market/index returns for benchmarking
    'crsp_msi': ("""
        select date, vwretd, vwretx, ewretd, ewretx, sprtrn
        from crsp.msi
        where date >= %(beg)s and date <= %(end)s
    """, ['date']),

    # CRSP-Compustat link, to attach fundamentals to a permno
    'ccm_linktable': ("""
        select gvkey, lpermno as permno, lpermco as permco, linktype, linkprim,
               linkdt, linkenddt
        from crsp.ccmxpf_lnkhist
        where linktype in ('LU','LC') and linkprim in ('P','C')
    """, ['linkdt', 'linkenddt']),

    # CRSP stock data stops at 2024-12-31 on this account while N-PORT runs to
    # 2026Q2, so Compustat's monthly security file covers the last 18 months.
    'comp_security': ("""
        select gvkey, iid, tic, cusip, isin, sedol, exchg, excntry, tpci, secstat
        from comp.security
    """, []),
    'comp_secm': ("""
        select gvkey, iid, datadate, tic, cusip, conm, prccm, cshoq, cshtrm,
               trt1m, trfm, ajexm, curcdm, exchg, secstat
        from comp.secm
        where datadate >= %(beg)s and datadate <= %(end)s
    """, ['datadate']),

    # Roughly 37% of equity positions in N-PORT are non-US, which CRSP and
    # Compustat North America do not cover. Compustat Global does, and matches on
    # the ISIN that N-PORT's IDENTIFIERS table carries for those securities.
    'comp_g_security': ("""
        select gvkey, iid, tic, cusip, isin, sedol, exchg, excntry, tpci, secstat
        from comp.g_security
    """, []),
    # annual fundamentals, trimmed to what portfolio-characteristic work needs
    'comp_funda': ("""
        select gvkey, datadate, fyear, indfmt, consol, popsrc, datafmt, curcd,
               at, lt, ceq, seq, ni, ib, oiadp, revt, sale, xrd, xsga, capx,
               dltt, dlc, che, csho, prcc_f, sich, naicsh
        from comp.funda
        where datafmt = 'STD' and indfmt = 'INDL' and consol = 'C' and popsrc = 'D'
          and datadate >= %(beg)s and datadate <= %(end)s
    """, ['datadate']),
}


def main():
    import wrds
    os.makedirs(DEST, exist_ok=True)
    user = tp.wrds_user()
    print('connecting to WRDS as %s ...' % (user or '(prompt)'), flush=True)
    db = wrds.Connection(wrds_username=user) if user else wrds.Connection()
    try:
        for name, (sql, dates) in QUERIES.items():
            if ONLY and name not in ONLY:
                continue
            out = os.path.join(DEST, name + '.parquet')
            if os.path.exists(out):
                print('  cached  %-20s %.0f MB' % (name, os.path.getsize(out) / 1e6), flush=True)
                continue
            t0 = time.time()
            try:
                params = {'beg': BEG, 'end': END} if '%(beg)s' in sql else None
                df = db.raw_sql(sql, params=params, date_cols=dates)
            except Exception as e:
                print('  FAILED  %-20s %s' % (name, str(e).splitlines()[0][:110]), flush=True)
                continue
            df.to_parquet(out, index=False)
            print('  pulled  %-20s %9d rows x %2d cols  %5.0f s  %6.0f MB'
                  % (name, len(df), df.shape[1], time.time() - t0,
                     os.path.getsize(out) / 1e6), flush=True)
    finally:
        db.close()
    print('\nparquet files in %s' % DEST)


if __name__ == '__main__':
    main()

"""Bridge holdings to gvkey on ISIN first, CUSIP second, and report what is lost.

Bridging on 9-digit CUSIP alone fails outside the US: only US listings carry one, so
Japan, Taiwan, Korea and the Netherlands matched at 0.0% and the names that dropped out
were TSMC, Samsung, SK Hynix and ASML - the AI hardware supply chain, which is precisely
the `enabler` leg under test. Attrition that deletes one arm of the hypothesis is not a
nuisance, it is a result-determining choice.

Order of resolution, most specific first:
    1. ISIN            -> gvkey, via Compustat Global + North America security tables
    2. CUSIP (9 or 8)  -> gvkey, via the same tables
    3. unmatched, carried as such

What matters downstream is not the row match rate but the share of each fund's PORTFOLIO
VALUE that reaches a firm measure, because a fund whose unmatched 30% is all AI chips is
mismeasured in a known direction. Both are reported, by fund and by country.
"""
import io, os

import numpy as np
import pandas as pd

from src import paths as tp  # noqa: E402

OUT = tp.out('holdings')
os.makedirs(OUT, exist_ok=True)
LOGS = tp.logs('holdings')
os.makedirs(LOGS, exist_ok=True)
MEAS = tp.out('holdings', 'firm_measures.parquet')
W = tp.data('wrds')


def gk(s):
    """Compustat gvkeys as six-digit strings."""
    return pd.to_numeric(s, errors='coerce').astype('Int64').astype(str).str.zfill(6)


def build_bridge(w):
    """isin -> gvkey and cusip -> gvkey from Compustat security tables."""
    frames = []
    for f, tag in (('comp_g_security.parquet', 'global'),
                   ('comp_security.parquet', 'na')):
        p = os.path.join(W, f)
        if not os.path.exists(p):
            continue
        d = pd.read_parquet(p)
        d.columns = [c.lower() for c in d.columns]
        d['gvkey'] = gk(d.gvkey)
        keep = {}
        for c in ('isin', 'cusip', 'excntry', 'tpci'):
            if c in d.columns:
                keep[c] = d[c]
        k = pd.DataFrame(keep)
        k['gvkey'] = d.gvkey
        k['src'] = tag
        frames.append(k)
    B = pd.concat(frames, ignore_index=True)
    isin = (B[B.get('isin', pd.Series(dtype=str)).notna()
              & (B.get('isin', pd.Series(dtype=str)).astype(str).str.len() > 5)]
            [['isin', 'gvkey']].drop_duplicates('isin'))
    cus = B[B.get('cusip', pd.Series(dtype=str)).notna()].copy()
    cus['cusip'] = cus.cusip.astype(str).str.strip().str.upper()
    cus = cus[cus.cusip.str.len() >= 8][['cusip', 'gvkey']].drop_duplicates('cusip')
    cus['cusip8'] = cus.cusip.str[:8]
    cus8 = cus.drop_duplicates('cusip8')[['cusip8', 'gvkey']]
    w('  bridge: %d ISINs, %d CUSIPs (%d distinct 8-char stems)'
      % (len(isin), len(cus), len(cus8)))
    return isin, cus, cus8


def main():
    """Bridge every equity holding to a gvkey, attach the firm measures and write
    holdings_bridged.parquet."""
    buf = io.StringIO()

    def w(s=''):
        """Print a line and keep it for the log file."""
        buf.write(s + '\n')
        try:
            print(s)
        except UnicodeEncodeError:
            print(s.encode('ascii', 'replace').decode())

    H = pd.read_parquet(os.path.join(OUT, 'holdings.parquet'))
    M = pd.read_parquet(MEAS)
    M['gvkey'] = gk(M.gvkey)

    w('=' * 96)
    w('BRIDGING HOLDINGS TO FIRMS')
    w('=' * 96)
    w('  equity rows : %d over %d funds' % (len(H), H.series_id.nunique()))
    isin, cus, cus8 = build_bridge(w)
    w()

    # Match on ISIN first, then on 9-character and 8-character CUSIP.
    H['isin_k'] = H['isin'].astype(str).str.strip().str.upper()
    H['cusip_k'] = H['cusip'].astype(str).str.strip().str.upper()
    H.loc[H.cusip_k.isin(['', 'N/A', 'NA', '000000000']), 'cusip_k'] = np.nan

    H = H.merge(isin.rename(columns={'isin': 'isin_k', 'gvkey': 'gv_isin'}),
                on='isin_k', how='left')
    H = H.merge(cus.rename(columns={'cusip': 'cusip_k', 'gvkey': 'gv_cusip'}),
                on='cusip_k', how='left')
    H['cusip8'] = H.cusip_k.str[:8]
    H = H.merge(cus8.rename(columns={'gvkey': 'gv_cusip8'}), on='cusip8', how='left')

    H['gvkey'] = H.gv_isin.fillna(H.gv_cusip).fillna(H.gv_cusip8)
    H['match_via'] = np.where(H.gv_isin.notna(), 'isin',
                              np.where(H.gv_cusip.notna(), 'cusip9',
                                       np.where(H.gv_cusip8.notna(), 'cusip8',
                                                'unmatched')))

    # Match rates by rows and by value, the ISIN-only gain, and the rate by country.
    tot = H.value.sum()
    w('-' * 96)
    w('MATCH RATE, BY ROW AND BY VALUE')
    w('-' * 96)
    w('  %-12s %10s %10s' % ('via', 'rows', 'value'))
    for k, g in H.groupby('match_via'):
        w('  %-12s %9.1f%% %9.1f%%' % (k, 100 * len(g) / len(H),
                                       100 * g.value.sum() / tot))
    w()
    w('  ISIN-ONLY GAIN: rows an ISIN reaches that no CUSIP would')
    only = H[(H.match_via == 'isin') & H.cusip_k.isna()]
    w('    %d rows, %.1f%% of all portfolio value' % (len(only),
                                                      100 * only.value.sum() / tot))
    top = only.groupby('issuer_name').value.sum().sort_values(ascending=False).head(10)
    for nm, v in top.items():
        w('      %-44s %5.2f%% of value' % (str(nm)[:44], 100 * v / tot))
    w()

    w('-' * 96)
    w('MATCH RATE BY INVESTMENT COUNTRY (value-weighted)')
    w('-' * 96)
    cv = (H.assign(matched=H.gvkey.notna())
           .groupby('country')
           .apply(lambda g: pd.Series({
               'share_of_value': g.value.sum() / tot,
               'matched': np.average(g.matched, weights=g.value.clip(lower=0) + 1e-9)}),
                  include_groups=False)
           .sort_values('share_of_value', ascending=False))
    w('  %-8s %14s %12s' % ('country', 'share of value', 'matched'))
    for c, r in cv.head(12).iterrows():
        w('  %-8s %13.2f%% %11.1f%%' % (c or '(blank)', 100 * r.share_of_value,
                                        100 * r.matched))
    w()

    # ---- does a firm measure actually attach? ----
    H = H.merge(M[['gvkey', 'conm', 'fic', 'sector', 'in_universe', 'pct_developer',
                   'pct_enabler', 'pct_adopter', 'pct_beneficiary',
                   'obs_developer', 'obs_enabler', 'obs_adopter', 'obs_beneficiary',
                   'has_developer', 'has_enabler', 'has_adopter',
                   'intense_developer', 'intense_enabler', 'intense_adopter',
                   'intense_beneficiary',
                   'is_enabler_core', 'is_enabler_power']], on='gvkey', how='left')

    w('-' * 96)
    w('DOES A MEASURE ATTACH? (value-weighted, this is the number that matters)')
    w('-' * 96)
    for c in ('developer', 'enabler', 'adopter', 'beneficiary'):
        m = H['obs_' + c].astype('boolean').fillna(False).astype(bool)
        w('  %-12s %6.1f%% of rows   %6.1f%% of portfolio value'
          % (c, 100 * m.mean(), 100 * H.loc[m, 'value'].sum() / tot))
    w()
    w('  A measure covering 40%% of FIRMS can still cover 85%% of VALUE, because the')
    w('  firms it covers are the large ones funds actually hold. The portfolio')
    w('  exposure (c01_portfolio_exposure) renormalises within the covered sleeve and carries the')
    w('  covered share alongside every number.')

    H.to_parquet(os.path.join(OUT, 'holdings_bridged.parquet'), index=False)
    w()
    w('wrote out/holdings_bridged.parquet (%d rows)' % len(H))

    with io.open(os.path.join(LOGS, 'h02_bridge.txt'), 'w', encoding='utf-8') as fh:
        fh.write(buf.getvalue())


if __name__ == '__main__':
    main()

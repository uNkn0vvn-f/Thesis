"""The patent CPC profile of an information set, capped at a filing year.

A profile without an upper bound would carry later filings into a label applied to earlier
holdings, so each information set counts only the patents filed from 2010 through UNTIL.
The settings (the top-N subclass composition, the titles) come from
src/firm_labels/e02_patent_profile.py.

    UNTIL      last filing year to include   (e.g. 2022 for the ChatGPT window)
    EVID_OUT   directory for patent_profile.parquet
"""
import io, os, sys

import pandas as pd

from . import e02_patent_profile as E2                      # noqa: E402  (CPC titles, paths)

UNTIL = int(os.environ['UNTIL'])
OUTDIR = os.environ['EVID_OUT']
os.makedirs(OUTDIR, exist_ok=True)


def main():
    """Write each universe firm's top CPC subclasses among patents filed from SINCE to
    UNTIL."""
    U = pd.read_parquet(E2.UNI, columns=['gvkey', 'name', 'value_all', 'value_claim'])
    U['gvkey'] = U.gvkey.astype(str)
    P = pd.read_parquet(E2.PAT, columns=['gvkey', 'primary_cpc_class', 'filing_year'])
    P['gvkey'] = P.gvkey.astype(str)
    P = P[(P.filing_year >= E2.SINCE) & (P.filing_year <= UNTIL)
          & P.gvkey.isin(set(U.gvkey))]
    assert int(P.filing_year.max()) <= UNTIL

    # Patent counts per firm and primary subclass, the TOPN largest kept.
    t = (P.groupby(['gvkey', 'primary_cpc_class']).size().rename('n').reset_index()
         .sort_values(['gvkey', 'n'], ascending=[True, False]))
    tot = t.groupby('gvkey').n.sum().rename('n_pat_since')
    top = t.groupby('gvkey').head(E2.TOPN)

    def fmt(d):
        """One firm's profile as "subclass count (title)" entries, largest first."""
        out = []
        for c, n in zip(d.primary_cpc_class, d.n):
            ttl = E2.CPC.get(c)
            out.append('%s %d%s' % (c, n, ' (%s)' % ttl if ttl else ''))
        return '; '.join(out)

    prof = top.groupby('gvkey').apply(fmt, include_groups=False).rename('cpc_profile')
    R = pd.concat([prof, tot], axis=1).reset_index()
    R = R.merge(U[['gvkey', 'name']], on='gvkey', how='left')
    R.to_parquet(os.path.join(OUTDIR, 'patent_profile.parquet'), index=False)
    msg = ('patents %d..%d: %d filings over %d firms (%.1f%% of held value); wrote %s'
           % (E2.SINCE, UNTIL, len(P), R.gvkey.nunique(),
              100 * U.loc[U.gvkey.isin(set(R.gvkey)), 'value_all'].sum()
              / U.value_all.sum(), os.path.join(OUTDIR, 'patent_profile.parquet')))
    print(msg)
    with io.open(os.path.join(OUTDIR, 'patent_profile.txt'), 'w', encoding='utf-8') as fh:
        fh.write(msg + '\n')


if __name__ == '__main__':
    main()

"""Which firms need an AI label, and why the existing measures are not enough.

Three proxy firm-level AI measures - USPTO G06N patents, Babina's resume-based
AI-employee share, and Eisfeldt's genAI occupational exposure - could be joined in a
union to ask whether an AI-named fund holds AI. That union is wrong at both ends, and the
error is not subtle:

  MISSES        SoundHound AI (SIC 7372, zero G06N patents, absent from Babina) is not
                flagged. Nor are BigBear.ai, Symbotic, Innodata, or Super Micro. A
                single-stock leveraged ETF on SoundHound therefore scores 0% AI with
                100% of its holdings measured.
  OVER-INCLUDES Walmart (36 G06N), Visa (52), Mastercard (8) and JPMorgan (14) are all
                flagged. 792 of 7,563 firms carry the flag, but they are 54.3% of total
                market capitalisation - so a fund making no AI claim at all already has
                a median 34% "AI" portfolio, and the 80% bar is unreachable by anyone.

The bottleneck in detecting AI washing is therefore the FIRM label, not the fund data.

THE UNIVERSE IS DEFINED BY VALUE HELD, NOT BY WHAT COMPUSTAT NORTH AMERICA COVERS.
Filtering to the firms present in `firm_measures`, which is built on Compustat North
America, would drop 5,229 held firms - NAURA Technology, Samsung SDI, Xinyi Solar, Wuxi
Lead - that are bridged by ISIN into Compustat GLOBAL. They are only 2.79% of value, but they include genuine AI
hardware names, and dropping the unmeasured part of the portfolio is exactly the blind
spot this step exists to remove. A further 3,019 held firms ARE in `firm_measures` but
carry `in_universe = False`, screened out of the measures for being investment vehicles or
having no positive revenue.

So the universe is assembled from the holdings side and metadata is attached from
whatever source has it:

    firms covering 99% of value held by funds making an AI claim
  u firms covering 95% of value held by every fund with holdings
  u every firm the existing union measure currently flags as AI

WHAT THE CLASSIFIER WILL HAVE TO WORK WITH, stated here because it is the design's
weakest point. Compustat carries no business description: only a company name, an
industry code, a country and size. So the model is being asked what it already knows
about the company, cued by the name and industry, rather than being asked to read a
document. That is a real difference from the fund-claim classifier, where the evidence block was the whole
input, and it has consequences:

  * the model may not recognise a firm at all, so the prompt asks it to say so, and
    unrecognised firms are reported separately rather than counted as "not AI";
  * its knowledge has a cutoff and is uneven across small and non-US firms;
  * it can invent a plausible business for a name it does not know, which is why the
    two-run agreement, the filing text and the patent profile matter more here than for
    the fund claims, not less.
"""
import io, os

import numpy as np
import pandas as pd

from src import paths as tp  # noqa: E402

OUT = tp.out('firm_labels')
LOGS = tp.logs('firm_labels')
os.makedirs(OUT, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)

HOLD = tp.out('holdings', 'holdings_bridged.parquet')
EXP = tp.out('holdings', 'fund_quarter_exposure.parquet')
MEAS = tp.out('holdings', 'firm_measures.parquet')

COVER_CLAIM = 0.99
COVER_ALL = 0.95


def main():
    buf = io.StringIO()

    def w(s=''):
        buf.write(s + '\n')
        try:
            print(s)
        except UnicodeEncodeError:
            print(s.encode('ascii', 'replace').decode())

    H = pd.read_parquet(HOLD, columns=['series_id', 'q', 'gvkey', 'value', 'conm',
                                       'issuer_name', 'country', 'fic', 'sector',
                                       'match_via'])
    H = H[H.value.notna() & (H.value > 0) & H.gvkey.notna()].copy()
    H['gvkey'] = H.gvkey.astype(str)
    X = pd.read_parquet(EXP, columns=['series_id', 'q', 'uses_ai', 'theme_broad'])
    for c in ('uses_ai', 'theme_broad'):
        X[c] = X[c].astype('boolean').fillna(False).astype(bool)
    X['claims'] = X.uses_ai | X.theme_broad
    J = H.merge(X[['series_id', 'q', 'claims']], on=['series_id', 'q'], how='left')
    J['claims'] = J.claims.fillna(False)

    w('=' * 96)
    w('THE FIRMS THAT NEED AN AI LABEL')
    w('=' * 96)
    w('  holding rows %d over %d distinct firms' % (len(J), J.gvkey.nunique()))
    w()
    w('-' * 96)
    w('HOLDINGS ARE CONCENTRATED, SO THE UNIVERSE IS DEFINED BY VALUE, NOT COUNT')
    w('-' * 96)
    w('  %-30s %10s %12s %12s %12s'
      % ('', 'firms', '50% of value', '90%', '99%'))
    for lab, sub in (('all funds with holdings', J),
                     ('funds making an AI claim', J[J.claims])):
        g = sub.groupby('gvkey').value.sum().sort_values(ascending=False)
        c = g.cumsum() / g.sum()
        w('  %-30s %10d %12d %12d %12d'
          % (lab, len(g), int((c <= .5).sum()) + 1, int((c <= .9).sum()) + 1,
             int((c <= .99).sum()) + 1))
    w()

    def cover(sub, th):
        g = sub.groupby('gvkey').value.sum().sort_values(ascending=False)
        c = g.cumsum() / g.sum()
        return set(g.index[:int((c <= th).sum()) + 1])

    s_claim = cover(J[J.claims], COVER_CLAIM)
    s_all = cover(J, COVER_ALL)

    M = pd.read_parquet(MEAS)
    M['gvkey'] = M.gvkey.astype(str)
    M['old_union'] = ((M.n_ai.fillna(0) > 0) | M.is_enabler_core.fillna(False)
                      | (M.bab_ai_share.fillna(0)
                         >= M.loc[M.bab_ai_share > 0, 'bab_ai_share'].quantile(.8)))
    s_old = set(M.loc[M.old_union & M.in_universe.fillna(False), 'gvkey'])
    U = (s_claim | s_all | s_old) & set(J.gvkey)

    w('-' * 96)
    w('THE UNIVERSE TO CLASSIFY')
    w('-' * 96)
    w('  %-46s %10s' % ('set', 'firms'))
    w('  %-46s %10d' % ('99% of value in AI-claiming funds', len(s_claim)))
    w('  %-46s %10d' % ('95% of value in all funds with holdings', len(s_all)))
    w('  %-46s %10d' % ('currently flagged AI by the union measure', len(s_old)))
    w('  %-46s %10d' % ('UNION, restricted to firms actually held', len(U)))
    w()
    for lab, sub in (('all funds', J), ('AI-claiming funds', J[J.claims])):
        v = sub.groupby('gvkey').value.sum()
        w('  covers %5.1f%% of value held by %s'
          % (100 * v[v.index.isin(U)].sum() / v.sum(), lab))
    w()

    # ---------------- assemble, from whatever source has it ----------------
    val = J.groupby('gvkey').value.sum().rename('value_all')
    valc = J[J.claims].groupby('gvkey').value.sum().rename('value_claim')
    base = (J.sort_values('value', ascending=False)
            .drop_duplicates('gvkey')
            .set_index('gvkey')[['conm', 'issuer_name', 'country', 'fic', 'sector',
                                 'match_via']])
    F = base[base.index.isin(U)].copy()
    MM = M.drop_duplicates('gvkey').set_index('gvkey')
    for c in ('sic4', 'revenue', 'mktcap', 'n_pat', 'n_ai', 'bab_ai_share',
              'eis_genai', 'is_enabler_core', 'is_enabler_power', 'old_union',
              'in_universe'):
        F[c] = MM[c].reindex(F.index) if c in MM.columns else np.nan
    for c in ('is_enabler_core', 'is_enabler_power', 'old_union', 'in_universe'):
        F[c] = F[c].astype('boolean').fillna(False).astype(bool)
    F['in_north_america'] = F.index.isin(set(M.gvkey))
    # the name the model sees: Compustat's if we have it, otherwise what the fund filed
    F['name'] = F.conm.fillna(F.issuer_name)
    F['ctry'] = F.fic.fillna(F.country)
    F = F.join(val).join(valc)
    F['value_claim'] = F.value_claim.fillna(0)
    F = F.sort_values('value_claim', ascending=False).reset_index()
    F['rank_claim'] = np.arange(1, len(F) + 1)

    w('-' * 96)
    w('WHERE THE METADATA COMES FROM')
    w('-' * 96)
    w('  %-40s %10s %14s' % ('', 'firms', '% of value held'))
    for lab, m in (('in Compustat North America', F.in_north_america),
                   ('bridged to Compustat GLOBAL only', ~F.in_north_america),
                   ('screened out of the measures (in_universe False)',
                    F.in_north_america & ~F.in_universe)):
        w('  %-40s %10d %13.1f%%'
          % (lab, int(m.sum()), 100 * F.loc[m, 'value_all'].sum() / F.value_all.sum()))
    w()
    w('  %-18s %10s' % ('field the model sees', 'non-null'))
    for c in ('name', 'ctry', 'sic4', 'sector', 'revenue', 'mktcap'):
        w('  %-18s %9.1f%%' % (c, 100 * F[c].notna().mean()))
    w()
    w('  A global firm has a name and a country but no SIC and no fundamentals, so the')
    w('  model gets less for exactly the firms it is least likely to recognise. That is')
    w('  a known weakness; `in_north_america` records it for every firm.')
    w()
    w('  the top 54 firms are half the value held by AI-claiming funds, so a hand check')
    w('  can start there and cover half the exposure in an afternoon.')
    w()

    F.to_parquet(os.path.join(OUT, 'firm_universe.parquet'), index=False)
    w('wrote out/firm_universe.parquet (%d firms)' % len(F))

    with io.open(os.path.join(LOGS, 'u01_build_universe.txt'), 'w',
                 encoding='utf-8') as fh:
        fh.write(buf.getvalue())


if __name__ == '__main__':
    main()

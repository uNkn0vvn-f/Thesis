"""Firm-level AI measures, screened so they mean what they are supposed to mean.

Four positions in the AI value chain are claimed by funds in src/fund_claims/. Three
have an externally built firm-level measure; the fourth has none and must come from
industry codes:

    developer    USPTO patents in CPC G06N - who actually builds AI technology
    enabler      NO EXTERNAL MEASURE EXISTS. Built here from SIC, and split in two:
                 enabler_core  semiconductors, fabrication equipment, servers,
                               storage, networking, components, data hosting
                 enabler_power electricity, transformers, switchgear, cooling
                 The split matters. Pooled, "enabler" would mean "semiconductors plus
                 every electric utility" - 190 of 403 firms would be utilities, most
                 of which have nothing to do with data centres - and the test would
                 return a result manufactured by the classification.
    adopter      Babina et al. (JFE 2024) AI-employee share from resumes - the firm
                 is hiring AI people, i.e. building AI capability in house
    beneficiary  Eisfeldt et al. generative-AI exposure - how much of the firm's
                 existing occupational mix genAI can do

BE PRECISE ABOUT THE LAST ONE. Eisfeldt's score is not "this firm benefits from AI". It
is the share of the firm's occupational mix that generative AI can perform, and whether
that is good or bad for the firm is exactly what their paper is about - their portfolio
is long high-exposure and short low-exposure, so they read it as value-relevant, but the
sign is a finding rather than a definition. After screening, the highest-scoring large
firms are IT services and software - CGI, Cognizant, Cerner, FIS, Snap, Xerox - whose
business is selling human hours that genAI can do. A fund saying it buys "companies that
benefit from AI" is not obviously saying it buys companies whose staff genAI can replace.
That is one reason 03 estimates the full claim x measure matrix instead of assuming a
mapping.

Construction of the three external measures is fixed, so
the counts are checkable against that log. What is new here is the SCREENING, and it is
not cosmetic. Unscreened, the top 15 US firms by Eisfeldt exposure are closed-end funds,
six Vanguard factor ETFs, a VoIP shell and two gas utilities; 39.3% of all firms
carrying an Eisfeldt score are SIC 67xx investment vehicles. The score is an
employment-weighted average over occupations, so an entity with almost no employees
lands wherever its handful of reported occupations happens to sit. Babina's tail is
polluted the same way by tiny and defunct firms where a few resumes make the share huge.

Screens, applied before any ranking:
    * drop SIC 67xx - investment offices, trusts, ETFs, closed-end funds
    * require a real operating business: positive revenue in the most recent
      Compustat fiscal year
    * rank within the screened universe as percentiles, not raw values, so that a
      measure with a narrow distribution (Eisfeldt: mean 0.37, sd 0.08) is comparable
      with one that is mostly zeros (Babina, patents)

NOT used, deliberately: an `ai_breadth` measure that counts how many
AI-named funds hold a stock. It is circular - the funds
being scored supply the definition. Everything here is built outside the fund universe.
"""
import io, os

import numpy as np
import pandas as pd

from src import paths as tp  # noqa: E402

OUT = tp.out('holdings')
LOGS = tp.logs('holdings')
os.makedirs(OUT, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)
W = tp.data('wrds')
EXT = tp.data('external_ai')


def gk(s):
    return pd.to_numeric(s, errors='coerce').astype('Int64').astype(str).str.zfill(6)


# --------------------------------------------------------------------------- #
# the enabler industry map: what AI runs ON, by SIC
# --------------------------------------------------------------------------- #
# Split in two, because pooling them would make "enabler" mean "semiconductors plus
# every electric utility". Utilities enter the AI story only insofar as they serve data
# centres, and SIC cannot tell which do; but funds like Tortoise AI Infrastructure,
# SoFi AI Power Grid and Defiance AI & Power Infrastructure explicitly claim the power
# leg, so it is reported rather than dropped.
#
# Note SIC is coarse and this is left uncorrected on purpose: Compustat files Apple
# under 3663, so the largest stock in the index lands in enabler_core. Hand-picking
# individual firms out would make the classification unreproducible, so instead the
# biggest members of each bucket are printed and the reader can see who is in it.
ENABLER_CORE_SIC = {
    3674: 'semiconductors',
    3672: 'printed circuit boards',
    3559: 'semiconductor fabrication equipment',
    3827: 'lab / optical instruments',
    3571: 'electronic computers / servers',
    3572: 'computer storage',
    3575: 'computer terminals',
    3576: 'computer network equipment',
    3577: 'computer peripherals',
    3661: 'telephone apparatus',
    3663: 'radio / TV / communications equipment',
    3669: 'communications equipment',
    3678: 'electronic connectors',
    3679: 'electronic components',
    7374: 'data processing and hosting',
}
ENABLER_POWER_SIC = {
    3585: 'refrigeration / cooling equipment',
    3612: 'power transformers / distribution',
    3613: 'switchgear / switchboard',
    3621: 'motors and generators',
    4911: 'electric services',
    4931: 'electric and other services',
}
ENABLER_SIC = {**ENABLER_CORE_SIC, **ENABLER_POWER_SIC}


def main():
    buf = io.StringIO()

    def w(s=''):
        buf.write(s + '\n')
        try:
            print(s)
        except UnicodeEncodeError:
            print(s.encode('ascii', 'replace').decode())

    # ---------------- developer: G06N patents ----------------
    p = pd.read_parquet(os.path.join(W, 'patents_permno_cpc.parquet'))
    p['gvkey'] = gk(p.gvkey)
    p['filing_year'] = pd.to_numeric(p.filing_year, errors='coerce')
    p['cpc'] = p.primary_cpc_class.astype(str).str.strip().str.upper()
    pre = p[p.filing_year.between(2013, 2022)]
    pat = pre.groupby('gvkey').agg(
        n_pat=('patent_num', 'size'),
        n_ai=('cpc', lambda s: (s == 'G06N').sum())).reset_index()
    pat['ai_patent_share'] = pat.n_ai / pat.n_pat

    # ---------------- adopter: Babina AI-employee share ----------------
    b = pd.read_stata(os.path.join(EXT, 'ai_firm_map_2021.dta'))
    b['gvkey'] = gk(b.gvkey)
    b = b[b.year == 2021].copy()
    b['bab_ai_share'] = np.where(b.totalempl > 0, b.aiempl / b.totalempl, np.nan)
    b['bab_totalempl'] = b.totalempl
    bab = b[['gvkey', 'bab_ai_share', 'bab_totalempl']].dropna(subset=['bab_ai_share'])

    # ---------------- beneficiary: Eisfeldt genAI exposure ----------------
    e = pd.read_csv(os.path.join(EXT, 'genaiexp_estz_firmscores.csv'))
    e['gvkey'] = gk(e.gvkey)
    eis = e[['gvkey', 'firmgenaiexp_estz_total']].rename(
        columns={'firmgenaiexp_estz_total': 'eis_genai'}).dropna()

    # ---------------- firm master ----------------
    comp = pd.read_parquet(os.path.join(W, 'comp_company.parquet'))
    comp['gvkey'] = gk(comp.gvkey)
    comp['sic4'] = pd.to_numeric(comp.sic, errors='coerce')

    def rng(lo, hi):
        return ((comp.sic4 >= lo) & (comp.sic4 <= hi)).fillna(False).to_numpy(bool)

    comp['sector'] = np.select(
        [rng(3570, 3579) | rng(3660, 3679) | rng(7370, 7379),
         rng(2830, 2836) | rng(8000, 8099), rng(6000, 6799)],
        ['tech', 'health', 'finance'], default='other')

    # revenue and size, latest fiscal year available
    fa = pd.read_parquet(os.path.join(W, 'comp_funda.parquet'))
    fa['gvkey'] = gk(fa.gvkey)
    fa = fa.sort_values(['gvkey', 'fyear'])
    last = fa.groupby('gvkey').tail(1)
    last = last.assign(mktcap=last.csho * last.prcc_f)
    size = last[['gvkey', 'fyear', 'revt', 'sale', 'at', 'mktcap']].rename(
        columns={'fyear': 'size_fyear'})

    M = (comp[['gvkey', 'conm', 'fic', 'sic', 'sic4', 'sector']]
         .merge(pat, on='gvkey', how='left')
         .merge(bab, on='gvkey', how='left')
         .merge(eis, on='gvkey', how='left')
         .merge(size, on='gvkey', how='left'))
    M['revenue'] = M.revt.fillna(M.sale)

    # ---------------- enabler flag ----------------
    M['enabler_industry'] = M.sic4.map(ENABLER_SIC)
    M['is_enabler_core'] = M.sic4.isin(ENABLER_CORE_SIC).fillna(False)
    M['is_enabler_power'] = M.sic4.isin(ENABLER_POWER_SIC).fillna(False)
    M['is_enabler'] = M.is_enabler_core | M.is_enabler_power

    # ---------------- screens ----------------
    M['is_investment_vehicle'] = M.sic4.between(6700, 6799)
    M['has_revenue'] = M.revenue.fillna(0) > 0
    M['in_universe'] = ~M.is_investment_vehicle & M.has_revenue

    w('=' * 96)
    w('FIRM-LEVEL AI MEASURES, SCREENED')
    w('=' * 96)
    w('  Compustat firms           : %d' % len(M))
    w('  with a G06N patent count  : %d  (%d have >=1 G06N)'
      % (M.n_ai.notna().sum(), int((M.n_ai > 0).sum())))
    w('  with a Babina AI share    : %d' % M.bab_ai_share.notna().sum())
    w('  with an Eisfeldt exposure : %d' % M.eis_genai.notna().sum())
    w('  in an enabler industry    : %d' % int(M.is_enabler.sum()))
    w()
    w('-' * 96)
    w('WHAT THE SCREENS REMOVE')
    w('-' * 96)
    w('  %-40s %8s %8s %8s' % ('', 'all', 'SIC 67xx', 'no revenue'))
    for nm, col in (('firms carrying an Eisfeldt score', 'eis_genai'),
                    ('firms carrying a Babina share', 'bab_ai_share'),
                    ('firms with a G06N patent', 'n_ai')):
        sub = M[M[col].notna()] if col != 'n_ai' else M[M.n_ai > 0]
        w('  %-40s %8d %7.1f%% %7.1f%%'
          % (nm, len(sub), 100 * sub.is_investment_vehicle.mean(),
             100 * (~sub.has_revenue).mean()))
    w()
    w('  surviving universe: %d firms (%.1f%% of Compustat)'
      % (int(M.in_universe.sum()), 100 * M.in_universe.mean()))
    w()

    # ---------------- winsorise, then rank inside the screened universe ------
    # Eisfeldt's tail survives the revenue>0 screen: the top is still Caribbean
    # Utilities and firms with $6-20m of revenue, because the score averages over a
    # handful of reported occupations when a firm has few. corr(score, log revenue) is
    # only -0.12, so this is sparse occupational data at small firms rather than a
    # size effect, and the character of the tail changes completely once revenue is
    # above ~$5bn. Two defences: winsorise at 1/99 (the project's own protocol for
    # compositional outcomes), and carry a reliability flag so 03 can run on firms
    # whose occupational mix is well measured. Portfolio aggregation is value-weighted,
    # which suppresses the residual pollution on its own.
    U = M[M.in_universe].copy()
    # Missing is not the same thing in all four measures, and treating it as one thing
    # would bias whichever claim group happens to hold less-covered firms.
    #   developer   a firm with NO patent record has no AI patents. That is a real zero,
    #               so n_ai is filled with 0 before ranking.
    #   enabler     SIC covers ~98% of firms; absent SIC is genuinely unknown.
    #   adopter     absence from Babina means UNOBSERVED, not zero AI staff: the file
    #               covers 3,556 firms built from resumes. Left missing.
    #   beneficiary absence from Eisfeldt is likewise unobserved. Left missing.
    # The two unobserved measures are therefore aggregated over their covered sleeve and
    # renormalised at the portfolio stage, with the covered share of value reported -
    # the sparse-measure rule.
    U['n_ai_filled'] = U.n_ai.fillna(0.0)
    lo, hi = U.eis_genai.quantile([.01, .99])
    U['eis_genai_w'] = U.eis_genai.clip(lo, hi)
    U['eis_reliable'] = U.revenue >= 1000          # $1bn, Compustat units are $m
    for src, dst in (('n_ai_filled', 'pct_developer'), ('bab_ai_share', 'pct_adopter'),
                     ('eis_genai_w', 'pct_beneficiary')):
        U[dst] = U[src].rank(pct=True) * 100
    U['pct_enabler'] = U.is_enabler_core.astype(float) * 100
    # ------------------------------------------------------------------ #
    # TWO-LAYER TREATMENT OF THE SPARSE MEASURES.
    # A percentile rank is the wrong transform for a variable that is zero for
    # 95.6% of firms: every firm in the zero block shares one mid-rank (47.8), so
    # Broadcom, SK Hynix, Arista and Vertiv - none of which holds a G06N patent -
    # score identically to a no-patent small cap, and Walmart (36 G06N patents)
    # outranks Broadcom. The sparse-measure rule is: "Split 'has the
    # attribute, top quintile of positive values' against 'zero'."
    #
    # So the sparse measures become, at firm level, an indicator and an intensity:
    #     has_*          the firm has the attribute at all
    #     intense_*      the firm is in the top quintile of POSITIVE values
    # and at portfolio level they aggregate to a share of value, which is what a
    # reader can actually interpret: "42% of this fund sits in firms holding AI
    # patents".
    # Eisfeldt is dense and near-normal, so it keeps the percentile.
    # ------------------------------------------------------------------ #
    U['has_developer'] = (U.n_ai_filled > 0)
    pos = U.loc[U.n_ai_filled > 0, 'ai_patent_share']
    U['intense_developer'] = (U.ai_patent_share.fillna(0)
                              >= pos.quantile(.8)) & U.has_developer
    U['has_adopter'] = U.bab_ai_share.fillna(0) > 0
    posb = U.loc[U.bab_ai_share > 0, 'bab_ai_share']
    U['intense_adopter'] = (U.bab_ai_share.fillna(0) >= posb.quantile(.8))
    U['has_enabler'] = U.is_enabler_core
    U['intense_enabler'] = U.is_enabler_core     # binary already
    # Eisfeldt is DENSE - every firm has some exposure - so a 'has the attribute'
    # indicator is meaningless for it; defining it
    # as the observation indicator would make it 100% by construction on the covered
    # sleeve. beneficiary keeps its percentile; only the sparse measures get the
    # two-layer treatment.
    U['intense_beneficiary'] = U.eis_genai_w >= U.eis_genai_w.quantile(.8)
    U['has_beneficiary'] = U.intense_beneficiary  # kept only so downstream columns exist

    U['obs_developer'] = True                     # a real zero is an observation
    U['obs_enabler'] = U.sic4.notna()
    U['obs_adopter'] = U.bab_ai_share.notna()
    U['obs_beneficiary'] = U.eis_genai.notna()
    twolayer = ['has_developer', 'intense_developer', 'has_enabler',
                'intense_enabler', 'has_adopter', 'intense_adopter',
                'has_beneficiary', 'intense_beneficiary']
    M = M.merge(U[['gvkey', 'n_ai_filled', 'eis_genai_w', 'eis_reliable',
                   'pct_developer', 'pct_adopter', 'pct_beneficiary', 'pct_enabler',
                   'obs_developer', 'obs_enabler', 'obs_adopter',
                   'obs_beneficiary'] + twolayer], on='gvkey', how='left')
    w('  two-layer sparse measures (share of screened firms):')
    for c in ('developer', 'enabler', 'adopter', 'beneficiary'):
        w('    %-12s has %5.1f%%   intense (top quintile of positives) %5.1f%%'
          % (c, 100 * U['has_' + c].mean(), 100 * U['intense_' + c].mean()))
    w()
    w('  Eisfeldt winsorised at 1/99: [%.3f, %.3f]; %d of %d screened firms have'
      % (lo, hi, int(U.eis_reliable.sum()), len(U)))
    w('  revenue >= $1bn and are flagged eis_reliable.')
    w()

    # ---------------- validation: does each measure now rank sensibly? --------
    w('-' * 96)
    w('VALIDATION - TOP 12 ON EACH MEASURE AFTER SCREENING')
    w('-' * 96)
    for lab, col, fmt, sub in (
            ('DEVELOPER  G06N patent count', 'n_ai', '%6.0f', U),
            ('ADOPTER    Babina AI-employee share', 'bab_ai_share', '%6.4f', U),
            ('EXPOSURE   Eisfeldt genAI exposure, revenue >= $1bn',
             'eis_genai', '%6.3f', U[U.eis_reliable]),
            ('EXPOSURE   Eisfeldt genAI exposure, revenue >= $5bn',
             'eis_genai', '%6.3f', U[U.revenue >= 5000])):
        w('  --- %s ---' % lab)
        for _, r in sub.nlargest(12, col)[['conm', 'fic', 'sector', col, 'revenue']].iterrows():
            w('     %-36s %-4s %-8s ' % (str(r.conm)[:36], r.fic, str(r.sector)[:8])
              + (fmt % r[col]) + '   rev %s'
              % ('%.0f' % r.revenue if r.revenue == r.revenue else '-'))
        w()
    for lab, msk in (('ENABLER-CORE  chips, compute, networking, hosting',
                      U.is_enabler_core),
                     ('ENABLER-POWER  electricity, transformers, cooling',
                      U.is_enabler_power)):
        w('  --- %s ---' % lab)
        for k, n in U[msk].enabler_industry.value_counts().head(8).items():
            w('     %-40s %5d firms' % (k, n))
        w('     largest by revenue:')
        for _, r in U[msk].nlargest(6, 'revenue')[
                ['conm', 'fic', 'enabler_industry', 'revenue']].iterrows():
            w('       %-30s %-4s %-34s %.0f' % (str(r.conm)[:30], r.fic,
                                                str(r.enabler_industry)[:34], r.revenue))
        w()
    w()
    w('  The $1bn list still carries oddities; only above ~$5bn does the ranking')
    w('  settle into IT services and software, whose business is selling the human')
    w('  hours genAI can do. Eisfeldt is therefore treated as a NOISY firm-level')
    w('  measure throughout: winsorised, aggregated value-weighted so that a $14m')
    w('  shell held at 0.1% weight contributes almost nothing, and run again on')
    w('  eis_reliable as a robustness check. No firm-level claim is made from it.')
    w()


    # ---------------- do the measures overlap? ----------------
    w('-' * 96)
    w('DO THE FOUR MEASURES PICK OUT THE SAME FIRMS?')
    w('-' * 96)
    U['top_dev'] = U.n_ai.fillna(0) > 0
    U['top_ado'] = U.bab_ai_share.fillna(0) >= U.loc[U.bab_ai_share > 0,
                                                     'bab_ai_share'].quantile(.8)
    U['top_ben'] = U.eis_genai >= U.eis_genai.quantile(.8)
    U['top_ena'] = U.is_enabler_core
    cols = ['top_dev', 'top_ena', 'top_ado', 'top_ben']
    names = ['developer', 'enabler', 'adopter', 'beneficiary']
    w('  Jaccard overlap between the "high" sets:')
    w('  %-14s %10s %10s %10s %10s' % ('', *names))
    for i, a in enumerate(cols):
        row = []
        for j, b2 in enumerate(cols):
            u = (U[a] | U[b2]).sum()
            row.append((U[a] & U[b2]).sum() / u if u else np.nan)
        w('  %-14s %10.3f %10.3f %10.3f %10.3f' % (names[i], *row))
    w()
    w('  set sizes: ' + ', '.join('%s %d' % (n, int(U[c].sum()))
                                  for n, c in zip(names, cols)))
    w()
    w('  Low overlap is the point, not a defect: these are four different questions')
    w('  about a firm, and a fund that claims one of them is not thereby claiming the')
    w('  others. If they picked out the same firms the claim-matching test would')
    w('  have nothing to separate.')

    M.to_parquet(os.path.join(OUT, 'firm_measures.parquet'), index=False)
    w()
    w('wrote out/firm_measures.parquet  (%d firms, %d in the screened universe)'
      % (len(M), int(M.in_universe.sum())))

    with io.open(os.path.join(LOGS, 'v01_build_measures.txt'), 'w',
                 encoding='utf-8') as fh:
        fh.write(buf.getvalue())


if __name__ == '__main__':
    main()

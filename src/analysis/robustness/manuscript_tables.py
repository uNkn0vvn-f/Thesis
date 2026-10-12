"""Format the manuscript's appendix tables directly from the robustness tables.

Writes each table as text (results/tables/robustness/manuscript_table_*.md) and a JSON of
every number quoted in the manuscript, so that the manuscript can be checked against the
saved outputs.

Tables A2 and A3 report the GPT-6 Luna labels (the thesis measurement) and, below them,
the DeepSeek labels (cross_model_correspondence.csv and robustness/deepseek/). The rows
taken from the second measurement were fixed before its estimates were computed: in A2 both
models' labels, each model's fund labels with the other's company labels, and the size-and-
sector adjusted DeepSeek row; in A3 the DeepSeek baseline; in A4 the DeepSeek baseline and
the DeepSeek annual-vintage design.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.analysis.robustness.common import TABLES, TABLES_DEEPSEEK  # noqa: E402

M = '−'


def f(x, d=1):
    s = ('%.' + str(d) + 'f') % x
    return s.replace('-', M)


def ci(r):
    return '%s [%s, %s]' % (f(r.avg_median_gap), f(r.ci_lo), f(r.ci_hi))


def effective_sample_sizes(table):
    """The A2 sample sizes: the complete-case OLS counts (ols_n, ols_funds).

    In the adjusted comparisons the only missing regression input is the exposure
    outcome, so these are also the nonmissing adjusted-outcome counts.
    """
    table = table.copy()
    for effective, saved in (('effective_fund_quarters', 'ols_n'),
                              ('effective_funds', 'ols_funds')):
        if effective not in table:
            if saved not in table:
                raise ValueError('A2 requires %s or %s' % (effective, saved))
            table[effective] = table[saved]
        if table[effective].isna().any():
            raise ValueError('A2 has missing %s' % effective)
    if (table.effective_fund_quarters.gt(table.fund_quarters).any()
            or table.effective_funds.gt(table.funds).any()):
        raise ValueError('A2 effective sample exceeds its eligible frame')
    return table


def adjusted_sample_notes(primary, secondary):
    """Disclosure of the adjusted outcomes' sample sizes against the bootstrap frames."""
    specs = [('GPT-6 Luna', primary, '5 size-and-sector adjusted outcome'),
             ('DeepSeek', secondary, 'J DeepSeek claims, DeepSeek labels, adjusted (full benchmark)')]
    parts, samples = [], {}
    for label, table, variant in specs:
        g = table[table.variant.eq(variant)].set_index('role')
        roles = ('developer', 'enabler', 'adopter')
        if set(g.index) != set(roles):
            raise ValueError('A2 needs all three adjusted roles for %s' % label)
        if g.funds.nunique() != 1 or g.fund_quarters.nunique() != 1:
            raise ValueError('A2 adjusted eligible frame differs by role for %s' % label)
        counts = g[['effective_fund_quarters', 'effective_funds']].astype(int)
        if len(counts.drop_duplicates()) == 1:
            n, funds = counts.iloc[0]
            count = '{:,} fund-quarters ({} funds) for each role'.format(n, funds)
        else:
            count = '; '.join('{}: {:,} fund-quarters ({} funds)'.format(
                role, *counts.loc[role]) for role in roles)
        frame = int(g.funds.iloc[0])
        parts.append('{}: {}; bootstrap frame {} eligible funds'.format(label, count, frame))
        samples[label] = g[['fund_quarters', 'funds', 'effective_fund_quarters',
                            'effective_funds']].astype(int).to_dict('index')
    note = ('Notes on sample sizes: The last column reports eligible fund-quarters (funds), '
            'before outcome missingness and any comparison-group restriction. For size-and-sector '
            'adjusted exposure, nonmissing outcomes are ' + '. '.join(parts) + '. '
            'The bootstrap resamples the eligible funds with their whole histories; missing adjusted '
            'outcomes are omitted from quarterly medians. Adjusted exposure uses only US-listed '
            'holdings with size-and-sector benchmark data. Both the usable fund-quarter sample and '
            'the holdings denominator can therefore differ from the raw comparison; changes in '
            'the gaps do not isolate benchmark subtraction.')
    return note, samples


def shift_share_table(nums):
    """Table A5: the event design's shift-share inference (event_shift_share.py), the GPT-6
    Luna and DeepSeek labels side by side. The rows were fixed before the Luna run was read."""
    J = {}
    for lab, d in (('luna', TABLES), ('deepseek', TABLES_DEEPSEEK)):
        p = d / 'shift_share.json'
        if p.exists():
            J[lab] = json.loads(p.read_text(encoding='utf-8'))
    if 'luna' not in J:
        return
    labs = [lab for lab in ('luna', 'deepseek') if lab in J]
    nums['shift_share'] = J
    head = {'luna': 'GPT-6 Luna labels', 'deepseek': 'DeepSeek labels'}
    en = ['| Specification | %s |' % ' | '.join(head[lab] for lab in labs), '|---|' + '---|' * len(labs)]

    def bse(b, se):
        return '%s (%s)' % (f(b, 3), f(se, 3))

    def pq(r):
        return '%s / %s' % (f(r['p_coef'], 3), f(r['p_t'], 3))

    def rec(lab, strata, est='recentred'):
        return J[lab]['recentred']['panel']['%s|%s' % (strata, est)]

    def add(le, cell):
        en.append('| %s | %s |' % (le, ' | '.join(cell(lab) for lab in labs)))

    add('*Stacked design, direction × exposure*', lambda lab: '')
    add('Fund-clustered',
        lambda lab: bse(rec(lab, 'none', 'raw')['b'], rec(lab, 'none', 'raw')['b'] / rec(lab, 'none', 'raw')['t']))
    add('Labels permuted within technology × size: mean, SD of the permuted estimates',
        lambda lab: '%s, %s' % (f(rec(lab, 'techsize', 'raw')['perm_mean'], 3), f(rec(lab, 'techsize', 'raw')['perm_sd'], 3)))
    add('  permutation p, coefficient / studentised', lambda lab: pq(rec(lab, 'techsize', 'raw')))
    for key, le in (('techsize', 'Recentred exposure, technology × size'),
                    ('sic2', 'Recentred exposure, two-digit industry'),
                    ('size', 'Recentred exposure, size quintile')):
        add(le, lambda lab, k=key: bse(rec(lab, k)['b'], rec(lab, k)['b'] / rec(lab, k)['t']))
        add('  permutation p, coefficient / studentised', lambda lab, k=key: pq(rec(lab, k)))
    add('Recentred, technology × size, expected exposure controlled',
        lambda lab: bse(rec(lab, 'techsize', 'rec+mu')['b'], rec(lab, 'techsize', 'rec+mu')['b'] / rec(lab, 'techsize', 'rec+mu')['t']))
    add('  permutation p, coefficient / studentised', lambda lab: pq(rec(lab, 'techsize', 'rec+mu')))
    if all('extra' in J[lab] for lab in labs):
        def ex(lab, k):
            return J[lab]['extra'][k]

        def fp(x):                      # the smallest attainable p is 1 / (draws + 1)
            return '<0.001' if x < 0.001 else f(x, 3)

        def p3(r):
            return '%s / %s / %s' % (fp(r['p_coef']), fp(r['p_t']), fp(r['p_per_sd']))
        add('Recentred, technology × size: permutation p per SD of exposure, raw / recentred',
            lambda lab: '%s / %s' % (fp(ex(lab, 'techsize|raw')['p_per_sd']), fp(ex(lab, 'techsize|recentred')['p_per_sd'])))
        add('Recentred, technology × size × holding-weight quintile',
            lambda lab: bse(ex(lab, 'techsize_hold|recentred')['b'], ex(lab, 'techsize_hold|recentred')['se']))
        add('  permutation p, coefficient / studentised / per SD',
            lambda lab: p3(ex(lab, 'techsize_hold|recentred')))
        add('Covered share × post and × direction controlled',
            lambda lab: bse(ex(lab, 'techsize|raw+cov')['b'], ex(lab, 'techsize|raw+cov')['se']))
        add('  recentred, technology × size',
            lambda lab: bse(ex(lab, 'techsize|recentred+cov')['b'], ex(lab, 'techsize|recentred+cov')['se']))
        add('  permutation p, coefficient / studentised / per SD',
            lambda lab: p3(ex(lab, 'techsize|recentred+cov')))

    def lo(lab, k):
        return J[lab]['leave_out'][k]
    add('Without NVIDIA', lambda lab: bse(lo(lab, 'without NVIDIA|b')['b'], lo(lab, 'without NVIDIA|b')['se']))
    add('Without the five largest AI stocks', lambda lab: bse(lo(lab, 'without the top 5|b')['b'], lo(lab, 'without the top 5|b')['se']))
    add('Net of the technology share', lambda lab: bse(lo(lab, 'nothing left out|b_tech')['b'], lo(lab, 'nothing left out|b_tech')['se']))
    add('Net of the technology share, without NVIDIA', lambda lab: bse(lo(lab, 'without NVIDIA|b_tech')['b'], lo(lab, 'without NVIDIA|b_tech')['se']))
    add('Placebo: pre-event months only, month −6 as the event',
        lambda lab: bse(J[lab]['inside_placebo']['full|placebo|stacked|b']['b'], J[lab]['inside_placebo']['full|placebo|stacked|b']['se']))
    add('*Single events, fund-level post-minus-pre means*', lambda lab: '')
    for tag, ts in (('chatgpt', 't ChatGPT'), ('deepseek', 't DeepSeek')):
        name = 'ChatGPT' if tag == 'chatgpt' else 'DeepSeek'
        add('%s (HC1 SE; shift-level SE)' % name,
            lambda lab, t=tag: '%s (%s; %s)' % (f(J[lab]['single'][t]['b'], 3), f(J[lab]['single'][t]['se_conventional'], 3),
                                                f(J[lab]['single'][t]['se_akm0'], 3)))
        add('  permutation p, studentised', lambda lab, k=ts: f(J[lab]['two_shock'][k]['p'], 3))
    add('Direction response (permutation SD)',
        lambda lab: bse(J[lab]['two_shock']['b']['observed'], J[lab]['two_shock']['b']['perm_sd']))
    add('  permutation p, coefficient / studentised',
        lambda lab: '%s / %s' % (f(J[lab]['two_shock']['b']['p'], 3), f(J[lab]['two_shock']['t b']['p'], 3)))
    add('DeepSeek placebo inside the pre-period (permutation SD)',
        lambda lab: bse(J[lab]['single']['deepseek']['placebo_b'], J[lab]['single']['deepseek']['placebo_perm_sd']))
    add('DeepSeek-day price shift: first-stage F; reduced form (t); IV (SE)',
        lambda lab: '%s; %s (%s); %s' % (f(J[lab]['single']['deepseek']['first_stage_F'], 1), f(J[lab]['single']['deepseek']['reduced_form_b'], 3),
                                         f(J[lab]['single']['deepseek']['reduced_form_t'], 2),
                                         bse(J[lab]['single']['deepseek']['iv_b'], J[lab]['single']['deepseek']['iv_se'])))
    (TABLES / 'manuscript_table_A5_en.md').write_text('\n'.join(en) + '\n', encoding='utf-8')


def main():
    R = effective_sample_sizes(pd.read_csv(TABLES / 'correspondence_inference.csv'))
    rows = [('1 baseline (coverage >= 50%)', 'Baseline (coverage ≥ 50%)'),
            ('2 coverage >= 30%', 'Coverage ≥ 30%'),
            ('2 coverage >= 90%', 'Coverage ≥ 90%'),
            ('3 evidence-only labels (coverage >= 50%)', 'Evidence-only company labels'),
            ('4a common sample, baseline labels', 'Same fund-quarters, baseline labels'),
            ('5 size-and-sector adjusted outcome', 'Size-and-sector adjusted exposure'),
            ('6 excluding the Magnificent Seven', 'Excluding the Magnificent Seven'),
            ("7 excluding each quarter's 10 most widely held", 'Excluding 10 most widely held per quarter'),
            ('8 versus theme funds naming no role', 'Versus AI-themed funds naming no role'),
            ('10 strict theme (AI a selection basis)', 'AI an explicit selection criterion'),
            ('11 excluding boilerplate/risk-only theme matches', 'Excluding risk-boilerplate theme matches')]
    en = ['| Specification | Developer | Enabler | Adopter | Eligible fund-quarters (funds) |', '|---|---|---|---|---|']
    nums = {}
    for key, le in rows:
        g = R[R.variant == key].set_index('role')
        cells = [ci(g.loc[r]) for r in ('developer', 'enabler', 'adopter')]
        n = '%s (%d)' % (format(int(g.fund_quarters.iloc[0]), ','), int(g.funds.iloc[0]))
        en.append('| %s | %s |' % (le, ' | '.join(cells + [n])))
        nums[key] = g[['avg_median_gap', 'ci_lo', 'ci_hi', 'p_boot', 'p_boot_holm', 'ols_coef', 'ols_se',
                       'ols_p', 'ols_p_holm', 'quarters_valid', 'quarters_positive', 'fund_quarters', 'funds',
                       'effective_fund_quarters', 'effective_funds']].round(4).to_dict('index')
    X2 = effective_sample_sizes(pd.read_csv(TABLES / 'cross_model_correspondence.csv'))
    xrows = [('B DeepSeek claims, DeepSeek labels (secondary)', 'DeepSeek fund and company labels'),
             ('C Luna claims, DeepSeek labels', 'DeepSeek company labels only'),
             ('D DeepSeek claims, Luna labels', 'DeepSeek fund labels only'),
             ('J DeepSeek claims, DeepSeek labels, adjusted (full benchmark)', 'DeepSeek labels, size-and-sector adjusted')]
    for key, le in xrows:
        g = X2[X2.variant == key].set_index('role')
        cells = [ci(g.loc[r]) for r in ('developer', 'enabler', 'adopter')]
        n = '%s (%d)' % (format(int(g.fund_quarters.iloc[0]), ','), int(g.funds.iloc[0]))
        en.append('| %s | %s |' % (le, ' | '.join(cells + [n])))
        nums['cross:' + key] = g[['avg_median_gap', 'ci_lo', 'ci_hi', 'p_boot', 'p_boot_holm', 'ols_coef', 'ols_se',
                                  'quarters_valid', 'quarters_positive', 'fund_quarters', 'funds',
                                  'effective_fund_quarters', 'effective_funds']].round(4).to_dict('index')
    # the inference rows: Holm-adjusted bootstrap p-values of the median gaps and the quarter-
    # effect OLS with its Holm-adjusted p-values, for the baseline and the size-and-sector
    # adjusted exposure, so that each claim in the text names its inference
    for key, le in (('1 baseline (coverage >= 50%)', 'baseline'),
                    ('5 size-and-sector adjusted outcome', 'size-and-sector adjusted')):
        hp = R[R.variant == key].set_index('role')
        en.append('| Holm-adjusted p of the median gaps, %s | %s | %s | %s | |' % ((le,) + tuple('%.3f' % hp.loc[r, 'p_boot_holm'] for r in ('developer', 'enabler', 'adopter'))))
        en.append('| OLS with quarter effects, %s (SE; Holm p) | %s | %s | %s | |' % ((le,) + tuple('%s (%s; %.3f)' % (f(hp.loc[r, 'ols_coef']), f(hp.loc[r, 'ols_se']), hp.loc[r, 'ols_p_holm']) for r in ('developer', 'enabler', 'adopter'))))
    note, samples = adjusted_sample_notes(R, X2)
    nums['A2:adjusted_samples'] = samples
    (TABLES / 'manuscript_table_A2_en.md').write_text('\n'.join(en) + '\n\n' + note + '\n', encoding='utf-8')

    H = pd.read_csv(TABLES / 'flow_contrast_sensitivity.csv').set_index('spec')
    A = pd.read_csv(TABLES / 'flow_contrast_age.csv').set_index('spec')
    T = pd.read_csv(TABLES / 'flow_contrast_inside_theme.csv')
    lg = json.loads((TABLES / 'flow_contrast_log.json').read_text(encoding='utf-8'))
    hrows = [('baseline, cluster series_id', H, 'Baseline, clustered by fund'),
             ('baseline, cluster sponsor', H, 'Clustered by registrant'),
             ('baseline, cluster fund+month', H, 'Two-way clustered by fund and month'),
             ('+ past 3- and 12-month returns (missing = indicator)', H, '+ past 3- and 12-month returns'),
             ('+ past returns + flow lags 2 and 3', H, '+ past returns and flow lags 2–3'),
             ('age bins for funds launched in panel + left-censored indicator', A, 'Age bins with left-censoring indicator'),
             ('funds launched within the panel only', A, 'Funds launched within the flow panel'),
             ('winsorise 5.0/95.0 within month', H, 'Winsorised at 5th/95th percentiles'),
             ('drop |raw flow| > 20% of assets', H, 'Excluding monthly flows above 20% of assets'),
             ('assets >= $50m', H, 'Assets of at least $50 million'),
             ('exposure-covered: + exposures + past returns', H, 'Holdings-covered funds, + AI and technology shares and past returns')]
    en = ['| Specification | Contrast (SE) | 95% interval | Fund-months | Funds |', '|---|---|---|---|---|']
    for key, src, le in hrows:
        r = src.loc[key]
        c = ['%s (%s)' % (f(r.r5_minus_r2, 2), f(r.se, 2)), '%s to %s' % (f(r.ci_lo, 2), f(r.ci_hi, 2)),
             format(int(r.n), ','), format(int(r.funds), ',')]
        en.append('| %s | %s |' % (le, ' | '.join(c)))
        nums['flow_contrast:' + key] = {'coef': float(r.r5_minus_r2), 'se': float(r.se)}
    lof, lor = lg['leave_one_out_fund'], lg['leave_one_out_registrant']
    en.append('| Leaving out any one fund: range of contrasts | %s–%s | | | |' % (f(lof['min'], 2), f(lof['max'], 2)))
    en.append('| Leaving out any one registrant: range of contrasts | %s–%s | | | |' % (f(lor['min'], 2), f(lor['max'], 2)))
    cb = lg['contrasts_baseline']['r5-r3']
    en.append('| Inside AI-themed funds: support for a named role vs no role named, baseline model | %s (%s) | %s to %s | | |' % (f(cb['coef'], 2), f(cb['se'], 2), f(cb['ci'][0], 2), f(cb['ci'][1], 2)))
    for fe, spec, le in (('month FE', '+ AI revenue share and technology share', 'Same, AI-themed funds only, month effects, + AI and technology shares'),
                         ('registrant x month FE', '+ AI revenue share and technology share', 'Same, AI-themed funds only, registrant-by-month effects')):
        r = T[(T.fe == fe) & (T.spec == spec)].iloc[0]
        en.append('| %s | %s (%s) | %s to %s | %s | %s |' % (le, f(r.r5_vs_r3, 2), f(r.r5_vs_r3_se, 2), f(r.r5_vs_r3 - 1.96 * r.r5_vs_r3_se, 2), f(r.r5_vs_r3 + 1.96 * r.r5_vs_r3_se, 2), format(int(r.n), ','), int(r.funds)))
    D3 = pd.read_csv(TABLES_DEEPSEEK / 'flow_contrast_sensitivity.csv').set_index('spec')
    r = D3.loc['baseline, cluster series_id']
    c = ['%s (%s)' % (f(r.r5_minus_r2, 2), f(r.se, 2)), '%s to %s' % (f(r.ci_lo, 2), f(r.ci_hi, 2)),
         format(int(r.n), ','), format(int(r.funds), ',')]
    en.append('| DeepSeek labels, baseline | %s |' % ' | '.join(c))
    nums['flow_contrast:deepseek baseline'] = {'coef': float(r.r5_minus_r2), 'se': float(r.se), 'r5_funds': int(r.r5_funds),
                                     'r5_fund_months': int(r.r5_fund_months), 'r2_funds': int(r.r2_funds),
                                     'r2_fund_months': int(r.r2_fund_months)}
    rb = H.loc['baseline, cluster series_id']
    nums['flow_contrast:groups'] = {'r5_funds': int(rb.r5_funds), 'r5_fund_months': int(rb.r5_fund_months),
                          'r2_funds': int(rb.r2_funds), 'r2_fund_months': int(rb.r2_fund_months)}
    P = pd.read_csv(TABLES / 'flow_contrast_premium.csv')
    P = P[P.spec == 'sponsor x month'].set_index('rung')
    nums['flow_contrast:versus no claim'] = {
        'theme with support (rung 5)': {'coef': float(P.loc[5, 'coef']), 'se': float(P.loc[5, 'se'])},
        'process claim (rung 2)': {'coef': float(P.loc[2, 'coef']), 'se': float(P.loc[2, 'se'])}}
    (TABLES / 'manuscript_table_A3_en.md').write_text('\n'.join(en) + '\n', encoding='utf-8')

    X = pd.read_csv(TABLES / 'event_extensions.csv')
    K = pd.read_csv(TABLES / 'event_fund_collapse.csv').set_index('event')

    def g(group, spec, ev):
        r = X[(X.group == group) & (X.spec == spec) & (X.event == ev)].iloc[0]
        return '%s (%s)' % (f(r.coef, 3), f(r.se, 3))
    erows = [(('A replication', 'baseline'), 'Baseline'),
             (('F timing', 'event month dropped'), 'Event month excluded'),
             (('F timing', 'window +/-6 months'), 'Six-month windows'),
             (('G controls', '+ past 3/12-month returns'), '+ past 3- and 12-month returns'),
             (('G controls', '+ both'), '+ past returns and flow lags 2–3'),
             (('D composition', 'balanced window'), 'Funds observed in every window month'),
             (('I annual vintages', 'baseline'), 'Annual company-label vintages, GPT-6 Luna labels')]
    en = ['| Specification | ChatGPT | DeepSeek | Stacked, direction × exposure |', '|---|---|---|---|']
    for (grp, spec), le in erows:
        st = 'balanced windows' if spec == 'balanced window' else spec
        cells = [g(grp, spec, 'chatgpt'), g(grp, spec, 'deepseek'), g(grp, st, 'stacked')]
        en.append('| %s | %s |' % (le, ' | '.join(cells)))
    XD = TABLES_DEEPSEEK / 'event_extensions.csv'
    if XD.exists():
        # the secondary labels (rows fixed before estimation)
        XD = pd.read_csv(XD)

        def gd(group, ev):
            r = XD[(XD.group == group) & (XD.spec == 'baseline') & (XD.event == ev)].iloc[0]
            return '%s (%s)' % (f(r.coef, 3), f(r.se, 3))
        for group, le in (('I annual vintages', 'Annual company-label vintages, DeepSeek labels'),
                          ('A replication', 'DeepSeek labels, baseline')):
            cells = [gd(group, 'chatgpt'), gd(group, 'deepseek'), gd(group, 'stacked')]
            en.append('| %s | %s |' % (le, ' | '.join(cells)))
    cells = [g('H industry', '+ industry shares x post (and tech)', 'chatgpt'), g('H industry', '+ industry shares x post (and tech)', 'deepseek'),
             g('H industry', '+ industry shares x post x sign (and tech)', 'stacked')]
    en.append('| + industry shares (semiconductors, software, hardware, electronics, communications, technology) | %s |' % ' | '.join(cells))
    en.append('| Fund-level post-minus-pre means | %s (%s) | %s (%s) | |' % (f(K.loc['chatgpt', 'coef_ai_pre'], 3), f(K.loc['chatgpt', 'se'], 3), f(K.loc['deepseek', 'coef_ai_pre'], 3), f(K.loc['deepseek', 'se'], 3)))
    (TABLES / 'manuscript_table_A4_en.md').write_text('\n'.join(en) + '\n', encoding='utf-8')
    shift_share_table(nums)
    for name in ('event_timing.json', 'event_pretrends.json', 'event_extensions_log.json'):
        p = TABLES / name
        if p.exists():
            d = json.loads(p.read_text(encoding='utf-8'))
            if name == 'event_timing.json':
                nums['event:pre_only_wald_p'] = {e: d[e]['p'] for e in ('chatgpt', 'deepseek')}
                nums['event:timing_sensitivity'] = d['timing_sensitivity']
            elif name == 'event_pretrends.json':
                nums['event:pre_event_wald_p_event_month_in_pre'] = {
                    e: v['pre_event_p'] for e, v in d['event_diagnostics'].items()}
            else:
                nums['event:date_scan'] = {k: v for k, v in d.items() if k.startswith('rolling_rank_')}
                nums['event:exposure_industry_corr'] = {k: v for k, v in d.items() if k.startswith('corr_ai_pre_with_industry_')}
    (TABLES / 'manuscript_numbers.json').write_text(json.dumps(nums, indent=2, default=str), encoding='utf-8')
    for p in ('A2', 'A3', 'A4', 'A5'):
        if (TABLES / ('manuscript_table_%s_en.md' % p)).exists():
            print((TABLES / ('manuscript_table_%s_en.md' % p)).read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()

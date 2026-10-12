"""Draw the flow-contrast regression estimates directly from the saved coefficient table.

    python src/analysis/plot_flow_contrast.py            GPT-6 Luna labels (the thesis measurement)
    THESIS_LABELS=deepseek python src/analysis/plot_flow_contrast.py

Reads flow_contrast_premium.csv written by src/analysis/robustness/flow_contrast_premium.py for the
selected labels and writes the figure, its caption with the numbers taken from the same
table, and the plotted values. Luna outputs go to results/figures/ and
results/tables/robustness/; DeepSeek outputs to their deepseek/ folders.
"""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.analysis.robustness.common import TABLES, FIGURES, LABELS  # noqa: E402
from src.analysis.paper_style import WIDTH, set_style, style_axes, save_figure  # noqa: E402

SOURCE = TABLES / 'flow_contrast_premium.csv'
OUT = FIGURES if LABELS == 'luna' else FIGURES / LABELS
OUT.mkdir(parents=True, exist_ok=True)
MODEL = {'luna': 'GPT-6 Luna', 'deepseek': 'DeepSeek'}[LABELS]

d = pd.read_csv(SOURCE)
d = d[d.spec.eq('sponsor x month')].set_index('rung')
process, theme, reverse = (d.loc[i] for i in [2, 5, -25])
contrast = -float(reverse.coef)
assert abs(contrast - (theme.coef - process.coef)) < 1e-8
se = float(reverse.se)
lo, hi = contrast - 1.96 * se, contrast + 1.96 * se
n, funds = int(reverse.n), int(reverse.funds)
m0, m1 = pd.Timestamp(reverse.month_start), pd.Timestamp(reverse.month_end)
plot_data = pd.DataFrame([
    {'term': 'Process-only AI claim vs no claim', 'coef': process.coef, 'se': process.se},
    {'term': 'AI theme with holdings support vs no claim', 'coef': theme.coef, 'se': theme.se},
    {'term': 'AI theme with holdings support minus process-only AI claim', 'coef': contrast, 'se': se}
])
plot_data['lower_95'] = plot_data.coef - 1.96 * plot_data.se
plot_data['upper_95'] = plot_data.coef + 1.96 * plot_data.se
plot_data.to_csv(TABLES / 'flow_contrast_figure.csv', index=False)
x_lo = min(-2.0, float(plot_data.lower_95.min()) - 0.4)
x_hi = max(4.4, float(plot_data.upper_95.max()) + 0.4)

period = '%s %d–%s %d' % (m0.strftime('%B'), m0.year, m1.strftime('%B'), m1.year)
set_style()
fig, (a, b) = plt.subplots(2, 1, figsize=(WIDTH, 4.10), sharex=True,
                           gridspec_kw={'height_ratios': [1.5, 1], 'hspace': .65})
fig.subplots_adjust(left=.305, right=.97, top=.93, bottom=.25)
for ax in [a, b]:
    style_axes(ax, grid='x')
    ax.axvline(0, color='#777777', linewidth=.65, linestyle=(0, (3, 3)))
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0, pad=9, labelsize=9)
    ax.set_xlim(x_lo, x_hi)
    ax.set_xticks([-2, 0, 2, 4])
# Shape and fill distinguish categories when printed without colour.
for row, y, marker, face in [(process, 1, 'o', '#111111'),
                              (theme, 0, 's', 'white')]:
    a.errorbar(row.coef, y, xerr=1.96 * row.se, fmt=marker,
               color='#111111', ecolor='#555555', markerfacecolor=face,
               markeredgewidth=.8, markersize=4.5, capsize=2.5, lw=.85)
a.set_ylim(-.55, 1.65)
a.set_yticks([1, 0], ['Process-only\nAI claim', 'AI theme with\nholdings support'])
for y, val in [(1, process.coef), (0, theme.coef)]:
    a.annotate(f'{val:+.2f}', (val, y), xytext=(0, 9), textcoords='offset points',
               ha='center', va='bottom', fontsize=9)
a.set_title('(a) Coefficients relative to no-claim funds', loc='left', fontsize=9.5, pad=8)
b.errorbar([contrast], [0], xerr=[[contrast - lo], [hi - contrast]], fmt='D',
           color='#111111', ecolor='#111111', markeredgewidth=.8,
           markersize=4.8, capsize=2.5, lw=.9)
b.set_ylim(-.5, .75)
b.set_yticks([0], ['Theme with support\nminus process-only'])
b.annotate(f'{contrast:+.2f}  [95% CI {lo:.2f}, {hi:.2f}]',
           (contrast, 0), xytext=(0, 10), textcoords='offset points',
           ha='center', va='bottom', fontsize=9)
b.set_title('(b) Theme with support minus process-only', loc='left', fontsize=9.5, pad=8)
b.set_xlabel('Monthly net flow rate (percentage points)', labelpad=7)
fig.text(.025, .082, 'Notes: Points are coefficients; whiskers show 95% normal-approximation intervals.',
         fontsize=8, ha='left', color='#555555')
fig.text(.025, .039, '%s; %s funds, %s fund-months; %s labels.' % (period, format(funds, ','), format(n, ','), MODEL),
         fontsize=8, ha='left', color='#555555')
save_figure(fig, OUT / 'flow_contrast')

caption = f"""Notes: Panel (a) reports two coefficients from the same registrant-by-month fixed-effects regression, relative to funds with neither an AI claim nor an AI name. Other disclosure categories remain in the regression. Process-only claims describe AI use in investment decisions without an AI investment theme. Holdings support requires an AI-themed fund to name a role and have positive size-and-sector-adjusted exposure to at least one named role in eligible, previously disclosed holdings. This does not establish support for every named role. Panel (b) reports the theme-with-support coefficient minus the process-only coefficient: {contrast:.2f} percentage points, with a fund-clustered standard error of {se:.2f}. Its interval uses the covariance of the two coefficients. The sample comprises {funds:,} funds and {n:,} fund-months from {period}. The outcome is net flow divided by beginning-of-month net assets, in percent, winsorised at the first and 99th percentiles within each month. Controls are log size, current return, prior-calendar-month flow and log one plus months since the first observation in the flow panel. Fund claims and company labels are from {MODEL}. Points are conditional regression estimates, not unadjusted group means or causal effects. Source: author's calculations from the disclosure and N-PORT panels.
"""
(OUT / 'flow_contrast_caption.md').write_text(caption, encoding='utf-8')

record = {'labels': LABELS, 'source': SOURCE.relative_to(ROOT).as_posix(),
          'specification': 'registrant x month (column spec: sponsor x month)',
          'contrast_direction': 'rung 5 minus rung 2',
          'estimate': contrast, 'clustered_se': se, 'lower_95': lo, 'upper_95': hi, 'n': n, 'funds': funds,
          'interval': 'estimate +/- 1.96 times the saved cluster-robust contrast SE; covariance is already included in that SE'}
(TABLES / 'flow_contrast_figure.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
print(json.dumps(record, indent=2))

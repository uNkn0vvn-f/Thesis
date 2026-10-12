"""Print-ready manuscript figures from saved result tables.

Run after ``python run.py``. Draws the four primary Luna figures without estimating
models or changing analytical tables. Outputs are high-resolution PNG and vector PDF.
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.analysis.robustness.common import TABLES, FIGURES, LABELS, load_claim_filings
from src.analysis.paper_style import (WIDTH, INK, MID, ROLE_COLORS, ROLE_MARKERS,
                                      ROLE_LINES, set_style, style_axes, save_figure)

CORE = ROOT / 'results' / 'tables'
SPECS = [('1 baseline (coverage >= 50%)', 'Baseline'),
         ('6 excluding the Magnificent Seven', 'Excluding Magnificent Seven'),
         ("7 excluding each quarter's 10 most widely held", 'Excluding 10 most\nwidely held companies'),
         ('3 evidence-only labels (coverage >= 50%)', 'Evidence-only labels'),
         ('5 size-and-sector adjusted outcome', 'Size-and-sector adjusted'),
         ('B DeepSeek claims, DeepSeek labels (secondary)', 'DeepSeek fund and\ncompany labels')]
CLAIM_STYLES = [
    ('any_ai_mention', 'Any AI mention', INK, '^', '-'),
    ('theme_broad', 'AI investment theme', MID, 's', '--'),
    ('uses_ai', 'AI in investment decisions', '#333333', 'o', '-.'),
]
# Group only mutually exclusive source categories; no observations are discarded.
COMPOSITION = [
    (('both',), 'Both', '#333333', ''),
    (('AI helps decide holdings',), 'AI process only', '#999999', '///'),
    (('buys AI companies',), 'AI theme only', '#dddddd', ''),
    (('may use AI (hedged)', 'AI for support tasks', 'AI risk disclosure only', 'AI keyword only, no claim'),
     'Other AI mentions', '#ffffff', '\\\\'),
]
ROLE_STYLES = [(role, role.capitalize(), ROLE_COLORS[role], ROLE_MARKERS[role], ROLE_LINES[role])
               for role in ROLE_COLORS] + [
    ('benefit_vague', 'Benefit mentioned; role not specified', '#444444', 'v', ':'),
    ('unspecified', 'Role unspecified', '#888888', 'D', (0, (5, 2, 1, 2))),
]


def save(fig, stem):
    save_figure(fig, FIGURES / (stem + '.png'))


def quarter_ticks(ax, qs):
    indices = sorted({0, len(qs) - 1} | {i for i, q in enumerate(qs) if str(q).endswith('q1')})
    ax.set_xticks(indices, [str(qs[i]).replace('q', 'Q') for i in indices])
    ax.set_xlim(-.35, len(qs) - .65)


def role_legend():
    return [Line2D([], [], color=ROLE_COLORS[r], marker=ROLE_MARKERS[r], ls=ROLE_LINES[r],
                   markerfacecolor='white', markeredgewidth=.7) for r in ROLE_COLORS]


def fig1():
    """Manuscript Figure 3: quarterly gaps and individual bootstrap intervals."""
    Q = pd.read_csv(TABLES / 'correspondence_quarterly_medians.csv')
    R = pd.read_csv(TABLES / 'correspondence_inference.csv')
    X = TABLES / 'cross_model_correspondence.csv'
    if X.exists():
        R = pd.concat([R, pd.read_csv(X)], ignore_index=True)
    specs = [s for s in SPECS if (R.variant == s[0]).any()]
    base = Q[Q.variant == SPECS[0][0]]
    fig = plt.figure(figsize=(WIDTH, 6.35))
    # Separate left margins keep specification labels out of the time-series panel.
    a = fig.add_axes([.105, .635, .87, .295])
    b = fig.add_axes([.365, .105, .61, .40])
    qs = sorted(base.q.unique())
    x = np.arange(len(qs))
    for role in ROLE_COLORS:
        t = base[base.role == role].set_index('q').reindex(qs)
        a.plot(x, t.gap, color=ROLE_COLORS[role], ls=ROLE_LINES[role],
               marker=ROLE_MARKERS[role], markerfacecolor='white', markeredgewidth=.7)
    a.axhline(0, color=MID, lw=.7)
    quarter_ticks(a, qs)
    a.set_ylabel('Exposure gap (pp)')
    a.set_title('(a) Quarterly median gaps', loc='left')
    a.legend(role_legend(), [r.capitalize() for r in ROLE_COLORS], ncol=3, loc='upper left')
    style_axes(a)
    for j, role in enumerate(ROLE_COLORS):
        for i, (key, _) in enumerate(specs):
            r = R[(R.variant == key) & (R.role == role)].iloc[0]
            y = i + (j - 1) * .22
            b.errorbar(r.avg_median_gap, y,
                       xerr=[[r.avg_median_gap - r.ci_lo], [r.ci_hi - r.avg_median_gap]],
                       fmt=ROLE_MARKERS[role], color=ROLE_COLORS[role],
                       mfc='white', mec=ROLE_COLORS[role], mew=.75,
                       ms=3.8, elinewidth=.8, capsize=2)
    b.axvline(0, color=INK, lw=.7, ls='--')
    b.set_yticks(range(len(specs)), [s[1] for s in specs])
    b.set_ylim(len(specs) - .5, -.55)
    b.set_xlabel('Average exposure gap (pp)')
    b.set_title('(b) Average gaps: individual 95% intervals', loc='left', fontsize=9)
    style_axes(b, 'x')
    b.spines['left'].set_visible(False)
    b.tick_params(axis='y', length=0, pad=7)
    if specs[-1][0].startswith('B DeepSeek'):
        b.axhline(len(specs) - 1.5, color=MID, lw=.6, ls=':')
    save(fig, 'role_correspondence')


def fig2():
    """Manuscript Figure 4: each specification uses its own saved standard errors."""
    R = pd.read_csv(TABLES / 'event_date_scan.csv')
    R['date'] = pd.to_datetime(R.date)
    R = R.sort_values('date')
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 5.15), sharex=True, sharey=True)
    fig.subplots_adjust(left=.115, right=.975, top=.93, bottom=.13, hspace=.42)
    bounds = []
    for a, (coef, se, title, marker) in zip(axes, [
        ('coef', 'se', '(a) Baseline controls', 'o'),
        ('coef_past_returns', 'se_past_returns', '(b) Adding past 3- and 12-month returns', 's'),
    ]):
        # Keep missing dates as gaps rather than joining across missing estimates.
        a.errorbar(R.date, R[coef], yerr=1.96 * R[se], fmt=marker + '-',
                   color=INK, ecolor='#999999', elinewidth=.55, capsize=1.5,
                   ms=2.7, mfc='white', mec=INK, mew=.6, lw=.8)
        bounds.extend([R[coef] - 1.96 * R[se], R[coef] + 1.96 * R[se]])
        a.axhline(0, color=MID, lw=.75)
        a.set_title(title, loc='left')
        a.set_ylabel('Flow coefficient (pp per pp)')
        style_axes(a)
        for d, label in [('2022-12-01', 'ChatGPT'), ('2025-02-01', 'DeepSeek')]:
            a.axvline(pd.Timestamp(d), color=MID, lw=.65, ls=':')
            a.text(pd.Timestamp(d), .98, label, transform=a.get_xaxis_transform(),
                   va='top', ha='center', fontsize=8, backgroundcolor='white')
    lo, hi = np.nanmin(np.concatenate(bounds)), np.nanmax(np.concatenate(bounds))
    for a in axes:
        a.set_ylim(lo - .06 * (hi - lo), hi + .12 * (hi - lo))
    ticks = sorted(set([R.date.min(), R.date.max()] + list(pd.date_range(R.date.min(), R.date.max(), freq='YS'))))
    axes[-1].set_xticks(ticks)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[-1].set_xlim(R.date.min() - pd.Timedelta(days=20), R.date.max() + pd.Timedelta(days=20))
    axes[-1].set_xlabel('First month treated as post-event')
    save(fig, 'event_date_scan')


def claims_by_year():
    """Manuscript Figure 1: rates, exclusive counts and overlapping roles."""
    A = pd.read_csv(CORE / 'claims_annual.csv')
    C = pd.read_csv(CORE / 'claims_composition.csv')
    V = pd.read_csv(CORE / 'claims_value_chain.csv')
    last = pd.to_datetime(load_claim_filings().date).max()
    years = A.year.to_numpy()
    x = np.arange(len(years))
    ytd = last.year == years.max() and (last.month, last.day) != (12, 31)
    xl = [str(y) + ('\nYTD' if ytd and y == years.max() else '') for y in years]
    fig, axes = plt.subplots(3, 1, figsize=(WIDTH, 7.3))
    fig.subplots_adjust(left=.11, right=.975, top=.955, bottom=.075, hspace=.47)
    a, b, c = axes
    for field, label, color, marker, line in CLAIM_STYLES:
        a.plot(x, A[field + '_rate_pct'], marker=marker, ls=line, color=color, lw=1,
               ms=3.5, mfc='white', mew=.65, label=label)
    a.set_ylabel('Share of funds (%)')
    a.set_ylim(bottom=0)
    a.set_title('(a) AI claim rates', loc='left')
    a.legend(loc='upper left', fontsize=8, labelspacing=.45, borderaxespad=.3)
    bottom = np.zeros(len(years))
    for keys, label, color, hatch in COMPOSITION:
        v = C[C.category.isin(keys)].groupby('year').funds.sum().reindex(years).fillna(0).to_numpy()
        b.bar(x, v, bottom=bottom, color=color, edgecolor=INK, hatch=hatch,
              lw=.35, width=.58, label=label)
        bottom += v
    expected = C.groupby('year').funds.sum().reindex(years).to_numpy()
    if not np.array_equal(bottom, expected):
        raise ValueError('The four displayed composition groups must retain all source counts.')
    for i, total in enumerate(bottom):
        b.text(i, total + bottom.max() * .015, format(int(total), ','), ha='center', fontsize=7.7)
    b.set_ylim(0, bottom.max() * 1.17)
    b.set_ylabel('Number of funds')
    b.set_title('(b) Composition of AI mentions', loc='left')
    handles, labels = b.get_legend_handles_labels()
    order = [2, 1, 0, 3]
    b.legend([handles[i] for i in order], [labels[i] for i in order], loc='upper left',
             ncol=2, fontsize=8, labelspacing=.45, borderaxespad=.3, handlelength=1.7)
    for key, label, color, marker, line in ROLE_STYLES:
        g = V[V.position.eq(key)].set_index('year').share_pct.reindex(years)
        c.plot(x, g, marker=marker, ls=line, color=color, lw=1, ms=3.5,
               mfc='white', mew=.65, label=label)
    c.set_ylim(0, 75)
    c.set_yticks([0, 20, 40, 60])
    c.set_ylabel('Share of AI-themed funds (%)')
    c.set_title('(c) Value-chain roles named', loc='left')
    c.legend(loc='upper left', ncol=2, fontsize=7.6, labelspacing=.3, borderaxespad=.2)
    for ax in axes:
        ax.set_xticks(x, xl)
        ax.set_xlim(-.45, len(years) - .55)
        style_axes(ax)
        if ytd:
            ax.axvspan(len(years) - 1.4, len(years) - .55, color='#eeeeee', zorder=0)
        if years.min() <= 2022 <= years.max():
            ax.axvline(np.interp(2022 + 11 / 12, years, x), color=MID, lw=.65, ls=':')
    save(fig, 'ai_claims_by_year')


def claimed_vs_held():
    """Manuscript Figure 2: separate fund prevalence and portfolio composition."""
    T = pd.read_csv(CORE / 'claimed_vs_held_quarterly.csv')
    qs = sorted(T.year.unique())
    xi = np.arange(len(qs))
    fig, axes = plt.subplots(3, 2, figsize=(WIDTH, 6.0), sharex=True, sharey=True,
                             gridspec_kw={'width_ratios': [1, 1.35]})
    fig.subplots_adjust(left=.105, right=.975, top=.89, bottom=.14, hspace=.5, wspace=.23)
    top = max(60, 10 * np.ceil(T[['claim_share', 'held_claimers', 'held_others']].max().max() / 10))
    for k, role in enumerate(ROLE_COLORS):
        g = T[T.position.eq(role)].set_index('year').reindex(qs)
        a, b = axes[k]
        a.plot(xi, g.claim_share, color=INK, ls='-', marker='o', ms=2.8, mfc='white', mew=.6)
        b.plot(xi, g.held_claimers, color=INK, ls='-', marker='o', ms=2.8, mfc=INK, mew=.6,
               label='AI-themed funds naming the role')
        b.plot(xi, g.held_others, color=MID, ls='--', marker='s', ms=2.8, mfc='white', mew=.6,
               label='Other AI-themed funds')
        for j, ax in enumerate((a, b)):
            ax.set_title('(%s) %s' % ('abcdef'[2*k+j], role.capitalize()), loc='left')
            ax.set_ylim(0, top)
            ax.set_yticks(np.arange(0, top + 1, 20))
            quarter_ticks(ax, qs)
            style_axes(ax)
        a.set_ylabel('AI-themed funds (%)')
        b.tick_params(axis='y', labelleft=True)
    fig.text(.29, .975, 'Role named in prospectus', ha='center', va='top', fontsize=10)
    fig.text(.755, .975, 'Median portfolio exposure (%)', ha='center', va='top', fontsize=10)
    indices = sorted({0, len(qs)-1} | {i for i, q in enumerate(qs) if str(q).endswith('q1') and int(str(q)[:4]) % 2 == 1})
    for ax in axes[-1]:
        ax.set_xticks(indices, [str(qs[i]).replace('q', 'Q') for i in indices])
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc='lower center', bbox_to_anchor=(.55, .025), fontsize=8)
    save(fig, 'claimed_vs_held')


def main():
    if LABELS != 'luna':
        raise SystemExit('The manuscript figures use Luna primary tables. Run with THESIS_LABELS=luna.')
    set_style()
    claims_by_year()
    claimed_vs_held()
    fig1()
    fig2()
    print('figures written to', FIGURES)


if __name__ == '__main__':
    main()

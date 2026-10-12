"""Print-safe supplementary figures for claims and holdings summaries."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .claims import COMPOSITION
from .holdings import POSITIONS
from .paper_style import (WIDTH, ROLE_COLORS, ROLE_LINES, ROLE_MARKERS,
                          save_figure, set_style, style_axes)


def _style() -> None:
    """Apply the shared figure style of paper_style."""
    set_style()


def _save_figure(fig, path: Path) -> None:
    """Save `fig` as PNG and PDF with paper_style.save_figure."""
    save_figure(fig, path)


def _line(ax, x, y, *, label, color="0.15", marker="o", ls="-", **kwargs):
    """A thin line with an open marker remains legible in monochrome print."""
    return ax.plot(x, y, label=label, color=color, marker=marker, ls=ls,
                   lw=1.05, ms=3.5, mfc="white", mec=color, mew=.7, **kwargs)


def claims(tables: dict[str, pd.DataFrame], folder: Path, latest_filing: str) -> None:
    """Three landscape panels emphasising the main claims and four exhaustive groups."""
    _style()
    annual = tables["claims_annual"]
    years = annual.year.to_numpy()
    x = np.arange(len(years))
    labels = [str(y) + ("\nYTD" if latest_filing[:4] == str(y)
                         and latest_filing[5:] != "12-31" else "") for y in years]
    fig, axes = plt.subplots(1, 3, figsize=(2 * WIDTH, 5.65))
    fig.subplots_adjust(left=.055, right=.985, top=.91, bottom=.40, wspace=.29)
    # Panel (a): claim rates by year.
    styles = [
        ("any_ai_mention", "Any AI mention", ".15", "^", "-"),
        ("theme_broad", "AI investment theme", ".30", "s", "--"),
        ("uses_ai", "AI in investment decisions", ".55", "o", "-."),
    ]
    for field, label, color, marker, line in styles:
        _line(axes[0], x, annual[field + "_rate_pct"], label=label,
              color=color, marker=marker, ls=line)
    axes[0].set_title("(a) AI claim rates", loc="left")
    axes[0].set_ylabel("Eligible funds (%)")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -.22), borderaxespad=0,
                   fontsize=8, labelspacing=.5, ncol=1, handlelength=2)

    # Panel (b): composition of the AI mentions, stacked.
    # Aggregate only the display; all seven source categories remain unchanged.
    groups = [
        ("Both", ["both"]),
        ("AI process only", ["AI helps decide holdings"]),
        ("AI theme only", ["buys AI companies"]),
        ("Other AI mentions", ["may use AI (hedged)", "AI for support tasks",
                               "AI risk disclosure only", "AI keyword only, no claim"]),
    ]
    fills = ["#333333", "#999999", "#dddddd", "#ffffff"]
    hatches = ["", "///", "", "\\\\"]
    bottom = np.zeros(len(years))
    composition = tables["claims_composition"]
    detail = composition.pivot(index="year", columns="category", values="funds").reindex(
        index=years, columns=COMPOSITION)
    if detail.isna().any().any():
        raise ValueError("Composition must contain all seven category counts for every year")
    for (name, categories), fill, hatch in zip(groups, fills, hatches):
        values = detail[categories].sum(axis=1).to_numpy()
        axes[1].bar(x, values, bottom=bottom, label=name, color=fill,
                    hatch=hatch, edgecolor=".15", lw=.35, width=.64)
        bottom += values
    if not np.array_equal(bottom, detail.sum(axis=1).to_numpy()):
        raise ValueError("Four displayed composition groups must preserve every annual total")
    for i, total in enumerate(bottom):
        axes[1].text(i, total + max(bottom) * .018, str(int(total)),
                     ha="center", va="bottom", fontsize=8)
    axes[1].set_title("(b) Composition of AI mentions", loc="left")
    axes[1].set(ylabel="Number of funds", ylim=(0, max(bottom) * 1.13))
    handles, names = axes[1].get_legend_handles_labels()
    order = [2, 1, 0, 3]
    axes[1].legend([handles[i] for i in order], [names[i] for i in order], loc="upper left",
                   bbox_to_anchor=(0, -.22), borderaxespad=0, fontsize=8,
                   handlelength=2, labelspacing=.45, ncol=2, columnspacing=1.0)

    # Panel (c): the named value-chain roles.
    chain = tables["claims_value_chain"]
    role_styles = [
        (role, role.capitalize(), ROLE_COLORS[role], ROLE_MARKERS[role], ROLE_LINES[role])
        for role in POSITIONS
    ] + [("benefit_vague", "Benefit mentioned, role not named", ".45", "v", ":"),
         ("unspecified", "Role unspecified", ".25", "D", (0, (5, 2, 1, 2)))]
    for role, label, color, marker, line in role_styles:
        group = chain[chain.position.eq(role)].set_index("year").reindex(years)
        _line(axes[2], x, group.share_pct, label=label, color=color, marker=marker, ls=line)
    axes[2].set_title("(c) Roles named in AI themes", loc="left")
    axes[2].set(ylabel="AI-themed funds (%)", ylim=(0, 100))
    axes[2].set_yticks(np.arange(0, 101, 20))
    axes[2].legend(loc="upper left", bbox_to_anchor=(0, -.22), borderaxespad=0,
                   fontsize=8, labelspacing=.5)
    # Shared year axis and the ChatGPT marker.
    for ax in axes:
        ax.set_xticks(x, labels)
        style_axes(ax)
        if years.min() <= 2022 <= years.max():
            transition = np.interp(2022 + 11 / 12, years, x)
            ax.axvline(transition, color=".55", ls=":", lw=.7, zorder=0)
    # The yearly counts of theme claimants go in an annotation rather than on a second axis.
    counts = "; ".join(f"{int(y)}: {int(n)}" for y, n in zip(years, annual.theme_broad))
    fig.text(.055, .123,
             "Other AI mentions: hedged AI use, support tasks, risk-only disclosure, or an AI keyword without a claim.",
             fontsize=8)
    fig.text(.055, .087, "AI-themed fund counts in panel (c) — " + counts + ".", fontsize=8)
    fig.text(.055, .051,
             f"SEC Form 485BPOS; one observation per fund-year; filings through {latest_filing}. "
             "Dotted vertical line: ChatGPT release.", fontsize=8)
    fig.text(.055, .020,
             "Final year is YTD when indicated. Annual flags use any filing; categories in (a) and roles in (c) may overlap. "
             "YTD filing populations can differ.", fontsize=8)
    _save_figure(fig, folder / "ai_claims_by_year_wide.png")


def _percentage_limit(values) -> float:
    """A y-axis limit in steps of 20, from 20 to 100, that fits the finite `values`."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return min(100., max(20., np.ceil(finite.max() / 20.) * 20.)) if len(finite) else 100.


def holdings(table: pd.DataFrame, frequency: str, folder: Path, latest_quarter: str) -> None:
    """Separate percentages of funds from percentages of holdings; preserve the gap panel."""
    _style()
    periods = sorted(table.year.unique())
    xs = {period: index for index, period in enumerate(periods)}
    fig = plt.figure(figsize=(2 * WIDTH, 5.85))
    grid = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.18], height_ratios=[1, 1.3],
                           left=.058, right=.99, top=.91, bottom=.29, wspace=.40, hspace=.43)
    claim_axes, held_axes = [], []
    claim_limit = _percentage_limit(table.claim_share)
    held_limit = _percentage_limit(table[["held_claimers", "held_others"]].to_numpy())
    gap_ax = fig.add_subplot(grid[:, 3])
    # Per role: the claim share (top row), the exposure of naming and other funds (bottom
    # row), and the gap in the right-hand panel.
    for index, position in enumerate(POSITIONS):
        claim = fig.add_subplot(grid[0, index])
        held = fig.add_subplot(grid[1, index], sharex=claim)
        claim_axes.append(claim)
        held_axes.append(held)
        group = table[table.position.eq(position)].set_index("year").reindex(periods)
        x = np.arange(len(periods))
        _line(claim, x, group.claim_share, label="AI-themed funds naming role", marker="D", ls=":")
        _line(held, x, group.held_claimers, label="AI-themed funds naming role", marker="s", ls="-")
        _line(held, x, group.held_others, label="Other AI-themed funds", color=".50", marker="o", ls="--")
        claim.set_title(f"({'abc'[index]}) {position.capitalize()}: claims", loc="left", fontsize=9.5)
        held.set_title(f"({'def'[index]}) {position.capitalize()}: holdings", loc="left", fontsize=9.5)
        claim.set_ylim(0, claim_limit)
        held.set_ylim(0, held_limit)
        claim.set_yticks(np.arange(0, claim_limit + 1, 20))
        held.set_yticks(np.arange(0, held_limit + 1, 20))
        claim.tick_params(axis="x", labelbottom=False)
        _line(gap_ax, x, group.gap, label=position.capitalize(),
              color=ROLE_COLORS[position], marker=ROLE_MARKERS[position],
              ls=ROLE_LINES[position])
    claim_axes[0].set_ylabel("AI-themed funds (%)")
    held_axes[0].set_ylabel("Classified holding value (%)")
    gap_ax.set_title("(g) Exposure gap:\nnaming minus other funds", loc="left", fontsize=9.5)
    gap_ax.set_ylabel("Percentage points")
    gap_ax.axhline(0, color=".45", lw=.65)
    gap_ax.legend(loc="upper left", fontsize=8)
    # Shared ticks, legend and notes.
    ticks = periods if frequency == "year" else [p for p in periods if str(p).endswith("q1")]
    ticklabels = [str(p)[:4] + ("\nQ" + latest_quarter[-1] if frequency == "year"
                  and str(p) == latest_quarter[:4] and latest_quarter[-1] != "4" else "")
                  for p in ticks]
    for ax in claim_axes + held_axes + [gap_ax]:
        ax.set_xticks([xs[p] for p in ticks], ticklabels)
        style_axes(ax)
    handles, names = held_axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="center", bbox_to_anchor=(.39, .18), ncol=2,
               title="Median portfolio exposure in panels (d)–(f)", title_fontsize=8)
    period_note = ("Each quarter uses its contemporaneous claim and holdings."
                   if frequency == "quarter" else
                   "Annual values: available-quarter mean within fund, then median across funds.")
    fig.text(.058, .098,
             f"AI-themed funds; classified coverage ≥ 50% of positive gvkey-mapped value; "
             f"at least five funds per comparison group; through {latest_quarter}.", fontsize=8)
    fig.text(.058, .061,
             "Panels (a)–(c): fraction of funds naming a role. Panels (d)–(f): median classified holding-value share. "
             "Scales are common within each row.", fontsize=8)
    fig.text(.058, .024, period_note + " Missing or suppressed observations leave gaps.", fontsize=8)
    _save_figure(fig, folder / f"claimed_vs_held_{'annual' if frequency == 'year' else 'quarterly'}_wide.png")


def workflow(folder: Path) -> None:
    """The fund and company measurement paths, using plain boxes and arrows."""
    from matplotlib.patches import FancyArrowPatch, Rectangle

    _style()
    fig, ax = plt.subplots(figsize=(WIDTH, 4.35))
    fig.subplots_adjust(left=.015, right=.985, top=.99, bottom=.15)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")

    def box(x, y, width, height, text):
        """A labelled box centred at (x, y)."""
        ax.add_patch(Rectangle((x - width / 2, y - height / 2), width, height,
                               facecolor="white", edgecolor=".15", lw=.65))
        ax.text(x, y, text, ha="center", va="center", fontsize=8, linespacing=1.4)

    def arrow(start, end):
        """An arrow from `start` to `end`."""
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8,
                                    lw=.7, color=".15", shrinkA=0, shrinkB=1.5))

    box(.16, .90, .28, .13, "SEC Form 485BPOS\nFund prospectus strategies")
    box(.50, .90, .27, .13, "SEC Form N-PORT\nQuarterly holdings")
    box(.84, .90, .28, .13, "Company filings and patents\nAnnual evidence vintages")
    box(.16, .64, .28, .14, "Fund claim labels\nProcess / theme / named roles")
    box(.84, .64, .28, .14, "Company role scores\nDeveloper / enabler / adopter")
    box(.50, .39, .34, .14, "Portfolio role exposures\nValue-weighted labelled holdings")
    box(.16, .12, .28, .14, "Figure 1\nWhat funds claim")
    box(.68, .12, .48, .14, "Figure 2\nClaimed versus held, quarterly")
    arrow((.16, .835), (.16, .71))
    arrow((.84, .835), (.84, .71))
    arrow((.50, .835), (.50, .46))
    arrow((.81, .57), (.65, .43))
    arrow((.16, .57), (.16, .19))
    ax.plot([.30, .315, .315, .50], [.64, .64, .24, .24], color=".15", lw=.7)
    arrow((.50, .24), (.55, .19))
    arrow((.55, .32), (.68, .19))
    fig.text(.04, .090, "Fund and company labels share the same value-chain role definitions.", fontsize=8)
    fig.text(.04, .049, "Fund claims are dated to prospectus filings; holdings use the corresponding annual labels.", fontsize=8)
    fig.text(.04, .012, "Figure 1 uses disclosures only. Figure 2 also uses holdings and company labels.", fontsize=8)
    _save_figure(fig, folder / "measurement_workflow.png")

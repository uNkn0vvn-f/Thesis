"""Compare stated AI value-chain roles with portfolio exposure."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import boolean, require


POSITIONS = ("developer", "enabler", "adopter")
MIN_FUNDS = 5


def period_statistics(source: pd.DataFrame, frequency: str, *,
                      min_funds: int = MIN_FUNDS, min_labelled_pct: float = 50
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Within fund: available-quarter mean; across funds: median above minimum n."""
    if min_funds < 1 or not 0 <= min_labelled_pct <= 100:
        raise ValueError("min_funds must be positive and min_labelled_pct must be within [0, 100]")
    if frequency not in {"year", "quarter"}:
        raise ValueError("frequency must be year or quarter")
    require(source, ["series_id", "q", "labelled", "theme_broad"]
            + list(POSITIONS) + ["chain_" + p for p in POSITIONS], "holdings")
    # Validate keys, quarters, flags and percentage ranges before anything is aggregated.
    data = source.copy()
    if data[["series_id", "q"]].isna().any().any() or data.duplicated(["series_id", "q"]).any():
        raise ValueError("holdings: require unique, nonmissing fund-quarter keys")
    if not data.q.astype(str).str.fullmatch(r"20\d{2}q[1-4]").all():
        raise ValueError("holdings: invalid quarter")
    data["theme_broad"] = boolean(data.theme_broad)
    for position in POSITIONS:
        data["chain_" + position] = boolean(data["chain_" + position])
    for col in ["labelled"] + list(POSITIONS):
        data[col] = pd.to_numeric(data[col], errors="raise")
        if ((data[col].notna()) & (~np.isfinite(data[col]) | ~data[col].between(0, 100 + 1e-9))).any():
            raise ValueError(f"holdings: {col} outside [0, 100] or infinite")
    # The comparison sample: AI-themed fund-quarters with enough classified holdings.
    data["period"] = data.q if frequency == "quarter" else data.q.str[:4].astype(int)
    selected = data.theme_broad & data.labelled.ge(min_labelled_pct)
    samples, rows = [], []
    for period, all_rows in data.groupby("period", sort=True):
        g = all_rows[selected.loc[all_rows.index]]
        samples.append({"frequency": frequency, "period": period,
                       "input_fund_quarters": len(all_rows),
                       "input_funds": all_rows.series_id.nunique(),
                       "non_theme_fund_quarters": int((~all_rows.theme_broad).sum()),
                       f"theme_below_{min_labelled_pct:g}_fund_quarters": int((all_rows.theme_broad & all_rows.labelled.lt(min_labelled_pct)).sum()),
                       "theme_missing_coverage_fund_quarters": int((all_rows.theme_broad & all_rows.labelled.isna()).sum()),
                       "selected_fund_quarters": len(g), "selected_funds": g.series_id.nunique(),
                       "period_meets_min_funds": g.series_id.nunique() >= min_funds})
        if g.series_id.nunique() < min_funds:
            continue
        # Per role: funds that name it against the other sampled funds, each fund reduced to
        # its mean exposure in the period, then the median across funds.
        for position in POSITIONS:
            fund = g.groupby("series_id", sort=True).agg(
                claim=("chain_" + position, "any"), held=(position, "mean"))
            claiming, other = fund[fund.claim], fund[~fund.claim]
            nc, no = len(claiming), len(other)
            nvc, nvo = int(claiming.held.notna().sum()), int(other.held.notna().sum())
            hc = float(claiming.held.median()) if nvc >= min_funds else np.nan
            ho = float(other.held.median()) if nvo >= min_funds else np.nan
            rows.append({"position": position, "year": period, "funds": len(fund),
                         "claim_share": 100 * nc / len(fund),
                         "held_claimers": hc, "held_others": ho, "gap": hc - ho,
                         "n_claimers": nc, "n_others": no,
                         "n_value_claimers": nvc, "n_value_others": nvo,
                         "fund_quarters": len(g)})
    cols = ["position", "year", "funds", "claim_share", "held_claimers", "held_others",
            "gap", "n_claimers", "n_others", "n_value_claimers", "n_value_others", "fund_quarters"]
    return pd.DataFrame(rows, columns=cols).sort_values(["position", "year"]).reset_index(drop=True), pd.DataFrame(samples)

"""Fund-year AI claims, stated value-chain positions and inclusion criteria."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import boolean, require


CHAIN = ["chain_developer", "chain_enabler", "chain_adopter",
         "chain_benefit_vague", "chain_unspecified"]
FLAGS = ["uses_ai", "uses_ai_hedged", "uses_ai_support", "theme_broad",
         "theme_strict", "theme_incidental", "chain_specified", "falsifiable_test",
         "strategy_text_risk_only"] + CHAIN
TEST_RANK = {"revenue threshold": 0, "index provider": 1,
             "manager judgement": 2, "none stated": 3}
COMPOSITION = ["both", "AI helps decide holdings", "buys AI companies",
               "may use AI (hedged)", "AI for support tasks",
               "AI risk disclosure only", "AI keyword only, no claim"]


def annual_statistics(source: pd.DataFrame, denominator: pd.DataFrame
                     ) -> tuple[dict[str, pd.DataFrame], dict]:
    """Any affirmative filing per year; roles overlap; strictest inclusion test wins."""
    require(source, ["date", "series_id", "series_resolved", "inclusion_test"] + FLAGS, "claims")
    require(denominator, ["year", "series_id"], "claims denominator")
    den = denominator[["year", "series_id"]].copy()
    den["year"] = pd.to_numeric(den.year, errors="raise").astype(int)
    if den.isna().any().any() or den.duplicated(["year", "series_id"]).any():
        raise ValueError("claims: denominator must have one nonmissing row per fund-year")
    fil = source.copy()
    date = pd.to_datetime(fil.date, errors="raise")
    resolved = boolean(fil.series_resolved)
    keep = date.notna() & resolved & fil.series_id.notna()
    counts_summary = {"input_filings": len(fil), "unresolved_filings": int((~resolved).sum()),
             "missing_date_filings": int(date.isna().sum()),
             "included_filings": int(keep.sum())}
    fil = fil[keep].copy()
    fil["year"] = date[keep].dt.year.astype(int)
    for flag in FLAGS:
        fil[flag] = boolean(fil[flag])
    if not fil.inclusion_test.isin(TEST_RANK).all():
        raise ValueError("claims: unknown or missing inclusion test")
    membership = fil[["year", "series_id"]].drop_duplicates().merge(
        den, on=["year", "series_id"], how="left", indicator=True)
    outside = membership.loc[membership._merge.ne("both"), ["year", "series_id"]]
    excluded_filings = fil.merge(outside, on=["year", "series_id"], how="inner")
    excluded = (excluded_filings.groupby(["year", "series_id"], sort=True)
                .agg(filings=("date", "size"), first_filing=("date", "min"),
                     last_filing=("date", "max")).reset_index())
    excluded["reason"] = "No resolved nonempty strategy in the eligible fund-year denominator"
    counts_summary["outside_denominator_fund_years"] = len(outside)
    counts_summary["outside_denominator_filings"] = len(excluded_filings)
    fil = fil.merge(den, on=["year", "series_id"], how="inner", validate="many_to_one")
    counts_summary["included_filings"] = len(fil)
    fil["own_any"] = fil.uses_ai | fil.uses_ai_hedged | fil.uses_ai_support
    # The composition definition requires that both claims must coexist in a filing.
    fil["both_claims"] = fil.uses_ai & fil.theme_broad
    flags = FLAGS + ["own_any", "both_claims"]
    fy = fil.groupby(["year", "series_id"], sort=True)[flags].any().reset_index()
    inclusion = (fil.assign(rank=fil.inclusion_test.map(TEST_RANK))
                 .groupby(["year", "series_id"], sort=True)["rank"].min())
    fy = fy.merge(inclusion.rename("inclusion_rank").reset_index(),
                  on=["year", "series_id"], validate="one_to_one")
    fy["inclusion_test"] = fy.inclusion_rank.map({v: k for k, v in TEST_RANK.items()})
    cross_filing = fy.uses_ai & fy.theme_broad & ~fy.both_claims
    counts_summary["cross_filing_combined_claim_fund_years"] = int(cross_filing.sum())
    # A nonzero result needs a scientific decision; keep the composition definition explicit.
    if cross_filing.any():
        raise ValueError("claims: claims coexist only across separate filings; check the composition definition")
    fy["category"] = "AI keyword only, no claim"
    fy.loc[fy.strategy_text_risk_only, "category"] = "AI risk disclosure only"
    fy.loc[fy.uses_ai_support & ~fy.uses_ai & ~fy.theme_broad, "category"] = "AI for support tasks"
    fy.loc[fy.uses_ai_hedged & ~fy.uses_ai & ~fy.theme_broad, "category"] = "may use AI (hedged)"
    fy.loc[fy.theme_broad & ~fy.uses_ai, "category"] = "buys AI companies"
    fy.loc[fy.uses_ai & ~fy.theme_broad, "category"] = "AI helps decide holdings"
    fy.loc[fy.both_claims, "category"] = "both"
    counts, composition, chain, tests = [], [], [], []
    for year, d in den.groupby("year", sort=True):
        f = fy[fy.year.eq(year)]
        th = f[f.theme_broad]
        n, nt = len(d), len(th)
        record = {"year": int(year), "funds_denominator": n, "any_ai_mention": len(f),
                  "any_ai_mention_rate_pct": 100 * len(f) / n}
        for flag in flags:
            record[flag] = int(f[flag].sum())
            record[f"{flag}_rate_pct"] = 100 * int(f[flag].sum()) / n
        counts.append(record)
        for cat in COMPOSITION:
            nc = int(f.category.eq(cat).sum())
            composition.append({"year": int(year), "category": cat, "funds": nc})
        for flag in CHAIN:
            nc = int(th[flag].sum())
            chain.append({"year": int(year), "position": flag.removeprefix("chain_"),
                          "theme_funds": nt, "claiming_funds": nc,
                          "share_pct": 100 * nc / nt if nt else np.nan})
        for test in TEST_RANK:
            nc = int(th.inclusion_test.eq(test).sum())
            tests.append({"year": int(year), "inclusion_test": test, "theme_funds": nt,
                          "funds": nc, "share_pct": 100 * nc / nt if nt else np.nan})
    return {"claims_annual": pd.DataFrame(counts),
            "claims_composition": pd.DataFrame(composition),
            "claims_value_chain": pd.DataFrame(chain),
            "claims_inclusion_rules": pd.DataFrame(tests),
            "claims_excluded_fund_years": excluded}, counts_summary


def denominator_summary(
    denominator: pd.DataFrame, history: pd.DataFrame, recovery: pd.DataFrame
) -> dict[str, int]:
    """Check eligible fund-year membership and unique-header series recovery."""
    require(history, ["date", "series_id", "series_resolved", "strategy_chars"], "claim history")
    require(denominator, ["year", "series_id"], "claims denominator")
    eligible = history[boolean(history.series_resolved) & history.series_id.notna()
                       & pd.to_numeric(history.strategy_chars, errors="raise").gt(0)]
    history_keys = set(zip(pd.to_datetime(eligible.date, errors="raise").dt.year.astype(int),
                           eligible.series_id))
    denominator_keys = set(zip(denominator.year.astype(int), denominator.series_id))
    if history_keys != denominator_keys:
        raise ValueError("claims: denominator differs from resolved nonempty-strategy claim history")
    require(recovery, ["year", "series_id", "strategy_chars", "strategy_sha256",
                       "header_series_count", "adds_fund_year_to_direct_denominator"], "denominator recovery")
    recovered_keys = set(zip(recovery.year.astype(int), recovery.series_id))
    if (not recovered_keys.issubset(denominator_keys)
            or not recovery.header_series_count.eq(1).all()
            or not recovery.strategy_chars.gt(0).all()
            or not recovery.strategy_sha256.str.fullmatch(r"[a-f0-9]{64}").all()):
        raise ValueError("claims: invalid unique-header series recovery")
    additions = recovery[boolean(recovery.adds_fund_year_to_direct_denominator)]
    added_keys = set(zip(additions.year.astype(int), additions.series_id))
    return {
        "denominator_fund_years": len(denominator_keys),
        "header_recovered_strategy_records": len(recovery),
        "header_recovery_additional_fund_years": len(added_keys),
    }

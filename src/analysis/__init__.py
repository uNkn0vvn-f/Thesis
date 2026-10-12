"""Run the research summaries from processed CSV inputs."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import claims, holdings, plots


SECTIONS = ("all", "claims", "holdings")
# The thesis measurement uses the GPT-6 Luna labels; the DeepSeek labels are the secondary
# measurement, written to tables/deepseek/ and figures/deepseek/.
LABELS = ("luna", "deepseek")
LABEL_INPUTS = {"luna": {"claim_filings": "claim_filings_luna",
                         "fund_quarter_exposure": "fund_quarter_exposure_luna"},
                "deepseek": {}}


def build(data_dir: Path, output_dir: Path, section: str = "all", labels: str = "luna") -> dict:
    """Write selected tables and figures, returning row counts for the run log."""
    if section not in SECTIONS:
        raise ValueError(f"section must be one of {SECTIONS}")
    if labels not in LABELS:
        raise ValueError(f"labels must be one of {LABELS}")
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    if data_dir.resolve() == output_dir.resolve() or data_dir.resolve() in output_dir.resolve().parents:
        raise ValueError("output_dir must not be inside the input directory")
    inputs: dict[str, pd.DataFrame] = {}
    tables: dict[str, pd.DataFrame] = {}
    summary: dict = {}

    def read(name: str) -> pd.DataFrame:
        frame = pd.read_csv(data_dir / f"{LABEL_INPUTS[labels].get(name, name)}.csv", low_memory=False,
                            dtype={"series_id": "string", "accession": "string"})
        inputs[name] = frame
        return frame

    if section in {"all", "claims"}:
        filings = read("claim_filings")
        denominator = read("claim_denominator_fund_year")
        history = read("claim_history")
        recovery = read("claim_denominator_recovery")
        summary["claims"] = claims.denominator_summary(denominator, history, recovery)
        claim_tables, claim_counts = claims.annual_statistics(filings, denominator)
        summary["claims"].update(claim_counts)
        tables.update(claim_tables)
        tables["claims_denominator_recovery"] = recovery
    if section in {"all", "holdings"}:
        exposure = read("fund_quarter_exposure")
        samples = []
        for frequency, suffix in [("year", "annual"), ("quarter", "quarterly")]:
            table, sample = holdings.period_statistics(exposure, frequency)
            tables[f"claimed_vs_held_{suffix}"] = table
            samples.append(sample)
        tables["holdings_sample"] = pd.concat(samples, ignore_index=True)
        summary["holdings"] = {"input_funds": int(exposure.series_id.nunique())}

    table_dir, figure_dir = output_dir / "tables", output_dir / "figures"
    if labels != "luna":
        table_dir, figure_dir = table_dir / labels, figure_dir / labels
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(table_dir / f"{name}.csv", index=False, lineterminator="\n", float_format="%.15g")
    if section in {"all", "claims"}:
        latest_filing = pd.to_datetime(inputs["claim_filings"].date, errors="raise").max()
        plots.claims(tables, figure_dir, latest_filing.strftime("%Y-%m-%d"))
    if section in {"all", "holdings"}:
        plots.holdings(tables["claimed_vs_held_quarterly"], "quarter", figure_dir,
                       str(inputs["fund_quarter_exposure"].q.max()))
    plots.workflow(figure_dir)
    # Only remove known, generated old outputs after their replacements have succeeded.
    retired = ["disclosure_trend.png"]
    if section in {"all", "claims"}:
        retired.append("claims_by_year.png")
    if section in {"all", "holdings"}:
        retired.extend(["claimed_vs_held_annual.png", "claimed_vs_held_quarterly.png"])
    for name in retired:
        (figure_dir / name).unlink(missing_ok=True)
    (table_dir / "disclosure_annual.csv").unlink(missing_ok=True)
    summary["input_rows"] = {name: len(frame) for name, frame in inputs.items()}
    summary["table_rows"] = {name: len(frame) for name, frame in tables.items()}
    return summary

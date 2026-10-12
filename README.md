# AI claims and fund holdings

Data processing and analysis of AI disclosures in US mutual-fund prospectuses and their
portfolio holdings: whether statements that a fund invests in AI developers, enablers or
adopters correspond to the fund's portfolio exposure to those activities, and how disclosure
categories and AI exposure relate to investor flows, including around the release of ChatGPT
and the DeepSeek-related repricing of AI stocks.

## Structure

```text
src/
  data/          SEC and WRDS data collection; processed-table export
  disclosure/    AI disclosure classification of the prospectuses
  fund_claims/   Strategy attribution, claim labels and dated fund panels
  holdings/      N-PORT extraction, firm matching and portfolio exposure
  firm_labels/   Filing and patent evidence; annual company labels
  analysis/      Claim trends, claim composition and claims versus holdings
    flows/       Monthly flow and return panels, base factors, AI factor, claim ladder,
                 flow regressions by claim type, alignment comparison, estimation helpers
    events/      ChatGPT and DeepSeek event analysis and its shift-share inference
    robustness/  Inference, sensitivity and cross-model analyses behind the manuscript
  paths.py       Shared project paths
  preparation.py Data-preparation workflow
data/
  raw/           Source downloads (local)
  interim/       Intermediate tables, model-response caches and analysis working files (local)
  processed/     CSV inputs used by the analysis
results/
  tables/        Analysis tables; robustness/ holds the manuscript's appendix tables
  figures/       Figures
docs/            Data fields, measurement definitions and the event design
Cited Papers/    Reading notes on the papers cited in the manuscript
run.py           Command-line entry point
requirements.txt Python dependencies
```

## Environment and commands

Python 3.13. Install the dependencies, then run the analysis:

```bash
python -m pip install -r requirements.txt
python run.py
python run.py --section claims
```

Sections: `claims`, `holdings`; the default runs both. Fund and company labels come from
two language models given the same prompts: OpenAI's GPT-6 Luna (the thesis measurement,
the default) and DeepSeek (`--labels deepseek`, written to `results/tables/deepseek/` and
`results/figures/deepseek/`).
The implementation uses pandas/NumPy for tables, Matplotlib for figures and PyArrow
for intermediate Parquet files. Source preparation also uses SciPy, requests, lxml and WRDS.

The repository supports three levels of reproduction:

| Level | What is available | Inputs needed |
|---|---|---|
| Core analysis from Git | `python run.py` and `python run.py --labels deepseek` regenerate the core tables and figures | Included `data/processed/` files and Python dependencies |
| Appendix presentation | Read the saved appendix tables, or redraw the four manuscript figures with `python src/analysis/robustness/figures.py` | Included processed inputs and saved `results/tables/`; this does not re-estimate regressions |
| Full appendix re-estimation | `python src/analysis/robustness/run_all.py` runs the appendix analyses on both label sets | External raw/interim files and cached model responses, listed in [data documentation](docs/data.md#inputs-of-the-robustness-analyses) |

The SEC and WRDS source files and the intermediate caches stay local (`data/raw/`,
`data/interim/`); [building from the sources](docs/data.md#building-from-the-sources) gives
the order in which the remaining inputs are built.

The data workflow is prospectus extraction → fund claims → holdings and firm labels →
portfolio exposure → processed CSVs → analysis. The preparation commands are:

```bash
python run.py prepare --list-steps
python run.py prepare
python run.py export
```

Preparation requires the local source files and caches in `data/raw/` and `data/interim/`;
its last steps build the monthly fund flow and return panels from the N-PORT and N-CEN
data sets and the monthly base factors (`src/analysis/flows/fund_month_panel.py`,
`base_factors.py`). Downloads are separate modules, listed with
their order in [building from the sources](docs/data.md#building-from-the-sources); the
SEC modules read a User-Agent with a contact address from `SEC_USER_AGENT` or the
git-ignored `.sec_user_agent` file.
`prepare --from STEP` resumes at a named stage. Model calls use cached responses by
default; `prepare --allow-api` enables requests using `DEEPSEEK_API_KEY`.
WRDS access requires an account with the relevant subscriptions; the username is read from
`WRDS_USER`, `.wrds_user` or the PostgreSQL password file.

Paths are defined relative to the project root in `src/paths.py`; no machine-specific
folders are required. `THESIS_DATA` and `THESIS_OUTPUT` (relative to the project root, or
absolute) redirect `data/raw/` and `data/interim/` for every module, the analyses in
`src/analysis/robustness/` included. New analyses belong in `src/analysis/` and write
tables and figures to `results/`.

## Figures and tables

`python run.py` writes the core tables to `results/tables/` and three figures, drawn by
`src/analysis/plots.py` from the current tables:

| Figure | Content |
|---|---|
| [`ai_claims_by_year_wide.png`](results/figures/ai_claims_by_year_wide.png) | Three panels: claim rates, composition and value-chain roles, by year |
| [`claimed_vs_held_quarterly_wide.png`](results/figures/claimed_vs_held_quarterly_wide.png) | Quarterly claims and holdings, one panel per role, and the claimant-minus-other gap |
| [`measurement_workflow.png`](results/figures/measurement_workflow.png) | The fund and company measurement paths |

The manuscript, kept outside the repository, rests on the analyses in [`src/analysis/robustness/`](src/analysis/robustness/README.md):
role comparisons with fund-cluster bootstrap intervals, the flow contrast by claim type, the
alignment comparison, the event analysis and its shift-share inference (label permutation,
recentred estimates, shift-level standard errors). After `python run.py`,
`python src/analysis/robustness/run_all.py` runs them on both label sets, writes their tables
to `results/tables/robustness/` (DeepSeek labels in `deepseek/`) and draws the four manuscript
figures to `results/figures/`. It checks each stage's dependencies,
records an execution manifest with input hashes, and exits with a non-zero code when an
input is missing or a stage fails. The [runner documentation](src/analysis/robustness/README.md)
describes partial runs and their manifests. The stratified agreement sample behind the
cross-model agreement table is drawn only on request (`run_all.py validation_sample`).

| Manuscript figure | File | Content |
|---|---|---|
| Figure 1 | `ai_claims_by_year.png` | The first figure above, stacked for the page width |
| Figure 2 | `claimed_vs_held.png` | Six panels: fund claim prevalence and median portfolio exposure in separate columns, one row per role |
| Figure 3 | `role_correspondence.png` | Quarterly median gaps and average gaps with bootstrap intervals |
| Figure 4 | `event_date_scan.png` | Candidate-date coefficients with pointwise intervals, separately for baseline and past-return controls |

All figures use the print-safe style in `src/analysis/paper_style.py`: serif English text,
black/grey marks distinguished by line type and marker shape, and matching 400 dpi PNG
and vector PDF exports. Figure 3's bootstrap intervals are individual, not adjusted for
multiple comparisons. To redraw the manuscript figures from existing tables without
rerunning regressions, use `python src/analysis/robustness/figures.py` with
`THESIS_LABELS=luna`.

`src/analysis/plot_flow_contrast.py` draws `results/figures/flow_contrast.png` and its
caption from `results/tables/robustness/flow_contrast_premium.csv`. The event
analysis uses company labels from event-specific historical information sets: the Luna
labels and exposures are built by `luna_measures.py`, the DeepSeek exposures by
`src.holdings.ai_exposure` on the cached pre-event labels (`LAB` and `EXP_OUT` settings).
The shift-share stage needs the CRSP daily file
(`TABLES=crsp_dsf_v2_2020_2025 python -m src.data.pull_wrds`).

See [data fields and sources](docs/data.md) and [measurement definitions and the event
design](docs/methods.md). [Cited Papers](Cited%20Papers/SUMMARY.md) lists the papers the
manuscript cites, with links.

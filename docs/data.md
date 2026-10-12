# Data

## Inputs

| Directory | Contents | Obtained by |
|---|---|---|
| `data/raw/edgar/` | EDGAR quarterly form indexes and the two lists of 485BPOS filings, 2020Q1 to 10 September 2026 | `python -m src.data.filing_index` |
| `data/raw/485bpos/` | Full text of every listed 485BPOS prospectus, one shard per month | `python -m src.data.download_485` |
| `data/raw/485bpos_headers/` | SEC series and class identifiers of each filing | `python -m src.data.fetch_headers` |
| `data/raw/485bpos_funds/` | Fund-level strategy text, split from the prospectuses | `python -m src.data.segment_funds` |
| `data/raw/nport/` | SEC Form N-PORT quarterly data sets | `python -m src.data.download_nport` |
| `data/raw/ncen/` | SEC Form N-CEN quarterly data sets: fund-type flags | `python -m src.data.download_nport` |
| `data/raw/wrds/` | Compustat, CRSP, Fama-French and USPTO patent-firm-link tables (licensed) | `python -m src.data.pull_wrds` |
| `data/raw/wrds/Loughran-McDonald_10X_Summaries_1993-2025.csv` | CIK, filing date and accession number of every 10-K | downloaded from the Notre Dame Software Repository for Accounting and Finance |
| `data/raw/external_ai/` | Babina et al. AI firm map (`ai_firm_map_2021.dta`) and the Eisfeldt, Schubert and Zhang generative-AI exposure scores (`genaiexp_estz_firmscores.csv`) | downloaded from the authors' data pages; the URLs are in `external_ai/dl.txt` |
| `data/raw/hand_input/` | `sibling_adjudications.csv`: definitions copied between sibling funds of a multi-fund prospectus, reviewed by hand | written by the author; read by `src.fund_claims.s01_extract_evidence` |

The SEC modules send the User-Agent of `src.paths.sec_user_agent()`, read from
`SEC_USER_AGENT` or the git-ignored `.sec_user_agent` file. Raw downloads are kept
unchanged. Processing writes to `data/interim/`, grouped by `disclosure`, `fund_claims`,
`holdings`, `firm_labels` and `exposure`. Cached model responses and the company evidence
sets also live there. `python run.py export` selects the analysis fields and writes
`data/processed/`. Source hashes are recorded in `data/processed/sources.json`.

## Processed tables

CSV files use UTF-8. Empty numeric and date fields mean missing, not zero. Read series
IDs, accessions and `gvkey` as strings. Percentages are stored on a 0–100 scale.

| File in `data/processed/` | Unit and main fields |
|---|---|
| `claim_filings.csv` | AI-bearing filing/series: resolved identity, AI-use/theme flags, value-chain roles, inclusion rule and agreement flags |
| `claim_denominator_fund_year.csv` | Eligible `year,series_id` pairs with nonempty strategy text |
| `claim_denominator_recovery.csv` | Strategy records assigned through a single-series SEC header; source file/line, strategy hash and whether the record adds a fund-year |
| `claim_history.csv` | Dated claims including explicit no-AI filings; strategy length and resolution status |
| `fund_label_agreement.csv` | Agreement statistics for each claim flag |
| `holdings_for_exposure.csv.gz` | Processed holding rows: series, quarter, firm ID, value and dates; identical retained fields can represent distinct holdings |
| `firm_label_vintages.csv` | Company/year role scores, both prompt responses and evidence-availability flags |
| `fund_quarter_exposure.csv` | Fund/quarter: holding counts and values, labelled coverage, role exposures and dated prospectus claims |
| `claim_filings_luna.csv`, `firm_label_vintages_luna.csv`, `fund_quarter_exposure_luna.csv` | The same tables on the GPT-6 Luna labels (the thesis measurement), built by `src/analysis/robustness/luna_measures.py` from the cached responses; the same company-years as the DeepSeek vintages, with evidence flags from them; the other files above carry the DeepSeek labels |

`port_value` is positive mapped holding value; `v_seen` is labelled value. `labelled`
is `100 * v_seen / port_value`. Role shares use labelled holding value as their denominator.
`date` is the matched prospectus date, distinct from the holdings `report_date` and
publication `filing_date`.

## Company label vintages

The label of calendar year Y reads only what was public by 31 December of Y-1: Item 1 of
the last 10-K filed in Y-1, or for a company without one, Item 4 of its last 20-F or the
annual information form of its last 40-F (`src.firm_labels.fetch_vintage_evidence`); its
patents filed through Y-1; and its last revenue and market capitalisation public by the
cutoff. Two DeepSeek prompts classify each company-year and `p03_label_pre` combines them.
The filing text of a 20-F or 40-F reaches the classifier under the same heading as 10-K
text ("Item 1 of its most recent Form 10-K"); the heading is part of the cached prompts of
both models and is therefore kept.

The classifier caches its answers per company (`firm_labels/vintages/01_classify/<year>/`).
In the 2026 vintage the second prompt's file holds two answers for 2,666 companies, written
by two runs of that prompt at the same time; the answers differ for 426 of them, and the
label takes the answer written last, as `p03_label_pre` does for every file.

## Analysis working folders

The regression analyses in `src/analysis/flows/`, `src/analysis/events/` and
`src/analysis/robustness/` keep their working files under `data/interim/analysis/<folder>/`.
Each folder is built by a module of this repository from the raw data, the pipeline's
interim outputs and the cached model responses; the first three rows are the last steps of
`python run.py prepare`, the rest follow the order given at the end of this page.

| Folder | Contents | Built by |
|---|---|---|
| `fund_flows` | Monthly N-PORT flow panel (928,506 fund-months, July 2019 to May 2026): gross sales, reinvestments and redemptions, the fund's return, the net flow (sales minus absolute redemptions, the definition that satisfies the net-asset identity), beginning-of-month assets reconstructed from that identity, and flow rates in percent | `src.analysis.flows.fund_month_panel`, from `data/raw/nport/` |
| `monthly_returns` | Monthly return panel (924,004 fund-months): the median monthly total return across a fund's share classes, the equal-weighted mean and class range, and the N-CEN fund-type flags; the month convention of the N-PORT fields is checked against the CRSP market return | the same module, from `data/raw/nport/`, `data/raw/ncen/` and `data/raw/wrds/crsp_msi.parquet` |
| `factors` | Monthly base factors: Carhart's four, compounded from the daily Fama-French file, and value-weighted enabler and technology industry portfolios net of the market | `src.analysis.flows.base_factors`, from `data/raw/wrds/ff_daily.parquet` and `crsp_msf_v2.parquet` |
| `flow_ladder`, `flow_ladder_luna` | The claim ladder (`src.analysis.flows.claim_ladder`) on the DeepSeek and the Luna labels | the ladder module (DeepSeek); `luna_measures.py` (Luna) |
| `claim_rungs_<labels>` | Flow regressions by claim type and their logs | `flow_contrast_premium.py` |
| `alignment_<labels>` | The alignment comparison: flows, and the alphas of the groups on Carhart's four factors and on Carhart plus the AI factor | `alignment_flows.py` |
| `alpha` | The AI factor: value-weighted returns of companies with at least half their revenue from AI and of the top decile of AI revenue share, net of the market, re-formed each January from the annual label vintages | `src.analysis.flows.ai_factor` |
| `preevent`, `preevent_luna` | Pre-event company labels and fund exposures of the event analysis, one information set per event: `00_evidence/<event>/` holds the business text (10-K, 20-F and 40-F filed in the window), the patent profile and the as-of financials, `01_classify/out/<event>/` the two DeepSeek classifier runs, `02_label/out/<event>/` the DeepSeek labels and `03_exposure/out/<event>/` the fund-quarter exposures | `src.firm_labels.run_preevent`; `luna_measures.py` builds the Luna labels and exposures from the cached `cross_model_labels.py` responses |
| `events` | Pre-event diagnostics of the event analysis and the exposure files of the non-AI technology control | `event_pretrends.py`, `event_timing.py` |
| `shift_share`, `shift_share_luna` | The shift-share inference on each label set; `shift_share/inputs/` holds the holdings file with the universe's sector that its modules read. Needs `data/raw/wrds/crsp_dsf_v2_2020_2025.parquet` | `event_shift_share.py` |
| `robustness`, `cross_model`, `validation` | Run manifests; the Luna responses and derived tables; the stratified agreement sample | `run_all.py`; `cross_model_labels.py`, `luna_measures.py`; `validation_sample.py` |

The pre-event information sets use the 10-K, 20-F and 40-F filings of 1 January to
30 November 2022 (ChatGPT) and of 1 January 2024 to 26 January 2025 (DeepSeek), patents filed
through 2022 and 2024, and revenue and market capitalisation selected with a fiscal-period-end
plus 90-day availability proxy. These are historical information vintages, not a guarantee
that every input was public before the event: the ChatGPT filing window includes the event
day, and patent filing years can precede publication. The companies are the 3,006 of the
classified universe when the sets were built (`firm_universe_preevent.parquet`, stored with
them); `u01_build_universe` now returns a slightly different universe because the holdings
data were extended afterwards. `00_evidence/SOURCE_SHA256.txt` records the hashes of the
stored sets and of that universe.

## Inputs of the robustness analyses

Core analysis uses the committed processed tables; saved appendix tables also support
inspection and figure redraws without regression estimation. Full appendix re-estimation
additionally requires the external raw/interim archive. The runner checks dependencies
for the requested stages and records their SHA256 in the execution manifest. Inputs
generated by an earlier requested stage are checked before their consuming stage runs.
`python src/analysis/robustness/run_all.py --check-inputs` checks the planned inputs
without running analyses or writing a manifest; stage names can narrow the check.
Availability is not proof of successful estimation. Execution manifests are stored in
`data/interim/robustness/runs/<run_id>/manifest.json`; `run_manifest.json` is only the
latest execution's snapshot. The dependencies come from four places:

- the processed CSVs of `python run.py export`, and the pipeline's interim outputs that the
  scripts read directly: `fund_claims/evidence.jsonl.gz` and `panel_fund_filing.parquet`
  (the filing text and the fund × filing claim panel), `holdings/holdings_bridged.parquet`,
  `stock_quarter.parquet` and `theme_universe.parquet`, `firm_labels/firm_universe.parquet`
  and `exposure/fund_quarter_exposure.parquet`;
- the monthly flow and return panels, the base factors, the AI factor and the claim ladder;
- the pre-event information sets, labels and exposures (`analysis/preevent/`) and the
  cached GPT-6 Luna responses in `cross_model/gpt-6-luna/` (`fund.jsonl`, `company.jsonl`,
  `company_preevent.jsonl`, the logged calls of `cross_model_labels.py`). The responses are
  the measurement itself: the model accepts no temperature setting, so a new run returns
  labels that differ in places, and the reported numbers are reproduced from these files
  rather than by calling the model again;
- the annual company-label vintages (`firm_labels/vintages/02_label/<year>/firm_ai_labels.parquet`),
  `fund_claims/labels_final_filing.parquet`, `fund_claims/ai_name_spells.parquet` and the
  attributed strategy shards `fund_claims/funds_attributed/*.jsonl.gz`, whose inventory
  `data/processed/sources.json` records;
- the WRDS files `ff_daily`, `crsp_dsf_v2_2020_2025`, `crsp_msf_v2` and `ccm_linktable`.

Raw datasets and local caches are excluded from Git. Vendor data retain their original
access and redistribution conditions.

The claim ladder is built from the monthly flows, the claim history and the exposure table.
The AI factor forms membership each January from the annual company-label vintages, local
CRSP/CCM inputs and the monthly base factors, and weights returns with the previous calendar
month's market capitalisation before filtering unavailable current returns. Claim-type flow
regressions require calendar-adjacent flow lags. The event baseline uses
the previous observed flow; its timing sensitivity additionally requires calendar
adjacency. For the claim ladder and holdings-controlled flow comparisons, holdings must
have been filed before the observation month and have a report date no more than 366 days
old; when several reports share a filing date, the latest report is used. The event
baseline uses its own pre-event selection rule and reports the 366-day restriction as a
sensitivity. An unidentified flow coefficient is left empty, rather than reported as a
zero effect.

## Building from the sources

`data/processed/sources.json` records the SHA256 of every source file that
`python run.py export` read, and the manifests of `run_all.py` record the hashes of its
inputs, so a rebuilt file can be checked against the reported one. The steps, in order:

| Order | Builder | Prerequisite |
|---|---|---|
| 1 | `python -m src.data.filing_index`, then `src.data.download_485`, `src.data.fetch_headers` and `src.data.segment_funds` | Network access to EDGAR and a User-Agent |
| 2 | `python -m src.data.download_nport`; `python -m src.data.pull_wrds` | A WRDS account with the Compustat, CRSP, Fama-French and patent-link subscriptions |
| 3 | The Loughran-McDonald file and the external AI measures in `data/raw/` (table above) | Downloaded by hand |
| 4 | `python run.py prepare` up to `u01_build_universe`, then `python -m src.firm_labels.fetch_vintage_evidence`, then `python run.py prepare --from "p02_patent_pre 2020"` | The cached DeepSeek classifications, or `--allow-api` |
| 5 | `python run.py export` | Step 4 |
| 6 | `python -m src.analysis.flows.claim_ladder`; `python -m src.analysis.flows.ai_factor` | Claims, name spells, monthly flows and exposure; the label vintages, base factors, CRSP and CCM |
| 7 | `python -m src.firm_labels.run_preevent` | `analysis/preevent/firm_universe_preevent.parquet` and the cached classifier runs in `analysis/preevent/01_classify/` |
| 8 | `python src/analysis/robustness/luna_measures.py --check-ladder` | The cached GPT-6 Luna responses and the DeepSeek inputs above |
| 9 | `python run.py`, then `python src/analysis/robustness/run_all.py` | Everything above |
| 10 | `python src/analysis/robustness/cross_model_labels.py compare --model gpt-6-luna` | The cached Luna responses and the agreement sample in `data/interim/validation/`; writes `cross_model_agreement.csv` without model calls |

`prepare` needs the vintage evidence fetched in step 4 only from its first `p02_patent_pre`
step onwards, so on a complete local copy it runs in one go. A new download or a new model
call does not reproduce the historical labels exactly; the cached responses preserve the
reported measurement. Source acquisition and WRDS access are separate from the code.

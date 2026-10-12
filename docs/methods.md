# Measurement definitions

## Disclosure and fund claims

The figures use the independent fund-claim flags of the classification layer.

The claim panel uses independent flags. Fund identity comes from attributed strategy
text or, for an unidentified block, a filing header containing exactly one series.
Annual numerator and denominator use the same eligible fund-year population. Annual
flags take `any` across filings. The `both` category requires AI use and an AI-theme
claim in the same filing. Inclusion rules take the most specific statement: revenue
threshold, index provider, manager judgement, none.

The claims figure has three panels: claim rates, composition of AI mentions and named
value-chain roles. It uses full calendar years
except the final year-to-date window, with the actual latest filing date shown. The
composition is exclusive; process/theme flags and role flags can overlap.

Developer, enabler and adopter roles can overlap. The fund and company classifiers share
the same role definitions.

Two models labelled the same texts with the same prompts. The thesis measurement uses
OpenAI's GPT-6 Luna, queried once with the first prompt wording; its API does not accept a
temperature setting, so a repeated query can return a different answer. DeepSeek's
deepseek-flash labels are the secondary measurement: its fund labels use the first
prompt's response, with a second wording measuring agreement. `run.py --labels deepseek`
and `THESIS_LABELS=deepseek` reproduce every table on the DeepSeek labels.

## Holdings

Positive holdings mapped to a company are matched to the corresponding annual label
vintage. A Luna company role is the first prompt's score; a DeepSeek role is the lower of
two prompt responses; a role counts at score 2 or 3. Portfolio exposure is the
value-weighted share within labelled mapped holdings. Coverage compares labelled value with
all positive mapped holding value, not total net assets. Both models label the whole classified
company universe and the event analysis's pre-event information sets.

The prospectus match is the most recent resolved filing on or before the portfolio
report date. No-AI filings remain in the history. Matches older than 487 days are
unknown. AI-theme comparisons require at least 50% labelled coverage.

Quarterly comparisons use each fund's contemporaneous claim and exposure. Annual
comparisons first take any claim and mean exposure within each fund, then take the
median across funds in each claim group. Each plotted median needs at least five
nonmissing fund-level exposures. A missing subgroup statistic leaves a gap, not zero.
The difference between groups is expressed in percentage points.

The claimed-versus-held figure is quarterly; annual summary tables remain available.
Both figures are calculated from the current CSV inputs. The workflow figure draws the
measurement paths. The manuscript's Figures 1 and 2 are the same two figures redrawn for
the page width; its Figure 2 omits the gap panel, which is Panel A of its Figure 3.

## Event design

The ChatGPT and DeepSeek event analysis treats a fund's pre-event AI-revenue exposure
as a shift-share variable: pre-event portfolio weights (shares) times the AI-revenue labels
of the companies held (shifts, from the pre-event information sets). Identification is
taken from the shifts, under the assumption that which companies carry higher labels is as
good as random within technology × size strata (size on 29 November 2022). Inference
follows Borusyak, Hull and Jaravel (2025): permutation of the company labels within strata
with studentised and coefficient-based p-values, the recentred estimator of Borusyak and
Hull (2023), exposure-robust standard errors from the equivalent company-level regression
(Adão, Kolesár and Morales, 2019) for the collapsed cross-sections, a covered-share control,
leave-out of the largest AI stocks, and a label-free price-based exposure for the DeepSeek
day. `src/analysis/robustness/event_shift_share.py` runs it on either label set; the
manuscript's Table A5 reports it. The single events are not identified separately.

## Interpretation

Disclosures measure claims and narratives. Agreement between prompts or between models
is not a human-coded accuracy measure; some company labels lack filing or patent evidence. Historical
input vintages cannot eliminate future information in model memory. Patent filing
year is not necessarily its public-availability date, and financial `datadate + 90 days`
is a timing proxy. Partial years, missing strategy text and unmapped holdings also
limit comparisons. These descriptive analyses do not establish causal effects.

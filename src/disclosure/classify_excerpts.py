r"""Classify the AI disclosures of fund prospectuses, 2020-2026, with DeepSeek.

The script reads the fund-level strategy shards (src.data.segment_funds), applies the AI
keyword screen, extracts compact keyword-context passages, and submits every candidate to
the five-category prompt of Chen, Sialm and Xu (2026, Internet Appendix IA.4.1), with the
wording adapted from adviser brochures (Form ADV) to fund prospectuses (Form 485BPOS).

The final CSV includes the complete fund-filing denominator.  Non-candidates are
explicitly marked ``not_sent_no_keyword``; candidate rows contain the exact text
sent to the API plus the returned category, quantitative flag, and evidence.
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests


from src import paths as tp  # noqa: E402

OUT_DIR = Path(tp.out("disclosure"))
FUND_DIR = Path(tp.data("485bpos_funds"))
API_KEY_FILE = Path(tp.API_KEY_FILE)
API_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-v4-flash"
PROGRESS = OUT_DIR / "02_deepseek_progress.jsonl"
CANDIDATES = OUT_DIR / "02_candidate_inputs.jsonl.gz"
PROFILE = OUT_DIR / "02_data_profile.json"
USAGE = OUT_DIR / "02_api_usage.json"
OUTPUT = OUT_DIR / "fund_classifications_2020_2026.csv"
LOG = OUT_DIR / "02_classification_log.txt"
MAX_EXCERPT_CHARS = 24_000

AI_KEYWORD = re.compile(
    r"artificial\s+intelligence|machine[\s-]?learning|deep\s+learning|neural\s+net|"
    r"natural\s+language\s+processing|large\s+language\s+model|generative\s+ai|"
    r"genetic\s+algorithm|evolutionary\s+comput|reinforcement\s+learning|chatgpt|"
    r"\bAI\b|\bA\.I\.|\bML\b|\bLLMs?\b|\bNLP\b",
    re.IGNORECASE,
)

CATEGORIES = {
    "AI-Autonomous",
    "AI-Augmented",
    "AI Support",
    "AI-Theme Assets",
    "AI Risk Disclosure",
    "No AI",
}

PROMPT = """Classify this mutual fund's AI/ML usage based on the Principal Investment Strategies text below, taken from its SEC Form 485BPOS statutory prospectus. Extract the following details:

1. AI category. Classify the fund's AI usage into one of the following categories:
AI-Autonomous: AI/ML is the primary investment decision-making engine - the strategy would not exist without AI/ML (e.g., fully systematic ML-driven trading, autonomous AI trading systems).
AI-Augmented: AI/ML is a significant input to investment decisions, but humans retain final decision-making authority (e.g., ML generates buy/sell signals that a portfolio manager reviews; NLP extracts sentiment used alongside fundamental analysis).
AI Support: The fund or its adviser uses AI/ML for research support or operational tasks that do not directly inform specific investment decisions - data extraction, dataset cleaning, report generation, compliance monitoring, client communication, back-office automation.
AI-Theme Assets: The fund invests in AI-themed stocks or AI-sector companies. The AI mention is about the investee companies, not about the fund's own investment process.
AI Risk Disclosure: AI is mentioned only in risk disclosures or disclaimers.
No AI: The text contains no genuine mention of AI/ML by the fund or its adviser (e.g. the keyword match was a false positive).

Assign exactly one category. When the text is consistent with more than one category, follow a strict priority order from AI-Autonomous > AI-Augmented > AI Support > AI-Theme Assets > AI Risk Disclosure.

2. Is quantitative? Does the fund use mathematical models, statistical techniques, algorithmic trading, systematic approaches, or any quantitative methodology as a core part of investment decisions? Any fund using AI/ML for investment decisions (AI-Autonomous or AI-Augmented) is inherently quantitative. Answer Y or N.

Return a json object with exactly these keys:
{"ai_category": "<one of: AI-Autonomous | AI-Augmented | AI Support | AI-Theme Assets | AI Risk Disclosure | No AI>", "is_quantitative": "<Y or N>", "evidence": "<the single most decisive verbatim quote from the disclosure, max 40 words>"}"""

FIELDNAMES = [
    "year",
    "filing_date",
    "accession",
    "registrant_cik",
    "registrant_name",
    "series_id",
    "series_name",
    "tickers",
    "n_classes",
    "n_strategy_blocks",
    "strategy_length_chars",
    "quant_keyword_in_strategy",
    "ai_keyword_candidate",
    "keyword_excerpt",
    "classification_status",
    "ai_category",
    "is_quantitative",
    "classification_evidence",
    "deepseek_model",
    "prompt_version",
    "api_input_sha256",
    "api_error",
]

_log_lock = threading.Lock()
_usage_lock = threading.Lock()
_usage = {"prompt_tokens": 0, "completion_tokens": 0, "api_calls": 0}


def log(message: str) -> None:
    """Print a timestamped message and append it to the classification log."""
    line = f"[{datetime.now():%H:%M:%S}] {message}"
    with _log_lock:
        print(line, flush=True)
        with LOG.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def iter_shards() -> tuple[list[dict], dict]:
    """The usable fund-filings of 2020-2026 from the fund-level shards, the last version of
    each (accession, series), with counts describing the input."""
    files = sorted(FUND_DIR.glob("funds_*.jsonl.gz"))
    if not files:
        raise FileNotFoundError(f"No fund shards found in {FUND_DIR}")
    rows: dict[tuple[str, str], dict] = {}
    stats = collections.Counter()
    signatures: dict[tuple[str, str], set[tuple]] = collections.defaultdict(set)
    for path in files:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                stats["raw_rows"] += 1
                row = json.loads(line)
                accession, series_id = row.get("accession"), row.get("series_id")
                if not accession or not series_id:
                    stats["unattributed_or_invalid_rows"] += 1
                    continue
                try:
                    year = int(str(row.get("date", ""))[:4])
                except ValueError:
                    stats["invalid_date_rows"] += 1
                    continue
                if not 2020 <= year <= 2026:
                    continue
                key = (accession, series_id)
                signatures[key].add(
                    (
                        row.get("date"),
                        int(row.get("strat_len") or 0),
                        bool(row.get("ai_strategy")),
                    )
                )
                rows[key] = row  # append-only shards: the last version is current
    stats["unique_accession_series"] = len(rows)
    stats["duplicate_rows"] = (
        stats["raw_rows"] - stats["unattributed_or_invalid_rows"] - len(rows)
    )
    stats["keys_with_changed_versions"] = sum(len(value) > 1 for value in signatures.values())
    usable = [row for row in rows.values() if int(row.get("strat_len") or 0) > 0]
    usable.sort(key=lambda row: (row["date"], row["accession"], row["series_id"]))
    stats["usable_fund_filing_rows"] = len(usable)
    stats["source_files"] = len(files)
    return usable, dict(stats)


def keyword_excerpt(text: str, radius: int = 1_200, cap: int = MAX_EXCERPT_CHARS) -> str:
    """The text around every AI keyword, `radius` characters on each side, nearby windows
    merged, capped at `cap` characters."""
    spans = [
        (max(0, match.start() - radius), min(len(text), match.end() + radius))
        for match in AI_KEYWORD.finditer(text)
    ]
    if not spans:
        return ""
    spans.sort()
    merged = [list(spans[0])]
    for start, stop in spans[1:]:
        if start <= merged[-1][1] + 200:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    excerpt = "\n\n[... next AI-keyword context ...]\n\n".join(
        text[start:stop].strip() for start, stop in merged
    )
    return excerpt[:cap]


def input_hash(excerpt: str, model: str) -> str:
    """SHA-256 of the model, the prompt and the excerpt: the cache key of one
    classification."""
    payload = f"{model}\n{PROMPT}\n--- FUND STRATEGY KEYWORD EXCERPT ---\n{excerpt}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def candidate_record(row: dict, model: str) -> dict:
    """The classifier input of one fund-filing: identifiers, excerpt and input hash."""
    excerpt = keyword_excerpt(row.get("strategy") or "")
    return {
        "accession": row["accession"],
        "series_id": row["series_id"],
        "input_sha256": input_hash(excerpt, model),
        "model": model,
        "filing_date": row["date"],
        "registrant_cik": row.get("cik"),
        "registrant_name": row.get("company"),
        "series_name": row.get("series_name"),
        "tickers": row.get("tickers") or [],
        "keyword_excerpt": excerpt,
    }


def write_candidates(candidates: list[dict]) -> None:
    """Write the candidate inputs to a gzipped JSON-lines file with a fixed gzip header."""
    with tp.gzip_text_writer(CANDIDATES) as handle:
        for row in candidates:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_progress() -> dict[tuple[str, str, str], dict]:
    """Successful answers in the progress ledger, keyed by accession, series and input hash."""
    output = {}
    if not PROGRESS.exists():
        return output
    with PROGRESS.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not row.get("error"):
                output[(row["accession"], row["series_id"], row["input_sha256"])] = row
    return output


def progress_call_counts() -> dict[str, int]:
    """Recover the call counts from the append-only progress ledger."""
    if not PROGRESS.exists():
        return {
            "progress_rows": 0,
            "base_decision_calls": 0,
            "identical_input_adjudication_calls": 0,
            "failed_result_rows": 0,
        }
    rows = []
    with PROGRESS.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return {
        "progress_rows": len(rows),
        "base_decision_calls": sum(
            not row.get("adjudicated_identical_input") and not row.get("error") for row in rows
        ),
        "identical_input_adjudication_calls": len(
            {
                row.get("input_sha256")
                for row in rows
                if row.get("adjudicated_identical_input") and not row.get("error")
            }
        ),
        "failed_result_rows": sum(bool(row.get("error")) for row in rows),
    }


def parse_category(value: object) -> str:
    """The answer's category mapped onto the five prompt categories, or PARSE_FAIL."""
    category = str(value or "").strip()
    if category in CATEGORIES:
        return category
    normalized = re.sub(r"[\s_-]+", " ", category.lower()).strip()
    for candidate in CATEGORIES:
        if re.sub(r"[\s_-]+", " ", candidate.lower()) in normalized:
            return candidate
    return "PARSE_FAIL"


def truncate_words(value: object, maximum: int = 40) -> str:
    """Enforce the paper prompt's evidence limit while retaining verbatim words."""
    words = str(value or "").split()
    return " ".join(words[:maximum])


def classify(candidate: dict, api_key: str, model: str, tries: int = 4) -> dict:
    """Send one excerpt to the model and return its category, quantitative flag and
    evidence, retrying rate limits and server errors."""
    body = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "user",
                "content": PROMPT
                + "\n\n--- FUND STRATEGY KEYWORD EXCERPT ---\n"
                + candidate["keyword_excerpt"],
            }
        ],
    }
    metadata = {
        "accession": candidate["accession"],
        "series_id": candidate["series_id"],
        "input_sha256": candidate["input_sha256"],
        "model": model,
    }
    for attempt in range(tries):
        try:
            response = requests.post(
                API_URL,
                json=body,
                timeout=180,
                headers={"Authorization": "Bearer " + api_key},
            )
            if response.status_code != 200:
                if response.status_code in {429, 500, 502, 503} and attempt < tries - 1:
                    time.sleep(3 * (attempt + 1))
                    continue
                return {**metadata, "error": f"HTTP {response.status_code}: {response.text[:300]}"}
            payload = response.json()
            answer = json.loads(payload["choices"][0]["message"]["content"])
            category = parse_category(answer.get("ai_category"))
            if category == "PARSE_FAIL":
                raise ValueError(f"Unrecognized category: {answer.get('ai_category')!r}")
            with _usage_lock:
                _usage["prompt_tokens"] += int(payload.get("usage", {}).get("prompt_tokens", 0))
                _usage["completion_tokens"] += int(
                    payload.get("usage", {}).get("completion_tokens", 0)
                )
                _usage["api_calls"] += 1
            return {
                **metadata,
                "ai_category": category,
                "is_quantitative": str(answer.get("is_quantitative", "")).strip().upper()[:1],
                "evidence": str(answer.get("evidence", ""))[:600],
                "error": None,
            }
        except Exception as error:  # noqa: BLE001 - retry and record boundary
            if attempt < tries - 1:
                time.sleep(3 * (attempt + 1))
                continue
            return {**metadata, "error": repr(error)[:500]}
    return {**metadata, "error": "retry loop exhausted"}


def export_csv(rows: list[dict], candidates: dict[tuple[str, str], dict], results: dict) -> None:
    """Write one row per usable fund-filing: identifiers, strategy facts, the excerpt sent
    and the classification, or why there is none."""
    temp = OUTPUT.with_suffix(".csv.tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            key = (row["accession"], row["series_id"])
            candidate = candidates.get(key)
            if candidate:
                result = results.get((key[0], key[1], candidate["input_sha256"]), {})
                status = "classified" if result and not result.get("error") else "api_error"
            else:
                result = {}
                status = "not_sent_no_keyword"
            writer.writerow(
                {
                    "year": row["date"][:4],
                    "filing_date": row["date"],
                    "accession": row["accession"],
                    "registrant_cik": row.get("cik") or "",
                    "registrant_name": row.get("company") or "",
                    "series_id": row["series_id"],
                    "series_name": row.get("series_name") or "",
                    "tickers": " ".join(row.get("tickers") or []),
                    "n_classes": row.get("n_classes") or 0,
                    "n_strategy_blocks": row.get("n_blocks") or 0,
                    "strategy_length_chars": row.get("strat_len") or 0,
                    "quant_keyword_in_strategy": bool(row.get("quant_strategy")),
                    "ai_keyword_candidate": candidate is not None,
                    "keyword_excerpt": candidate["keyword_excerpt"] if candidate else "",
                    "classification_status": status,
                    "ai_category": result.get("ai_category", ""),
                    "is_quantitative": result.get("is_quantitative", ""),
                    "classification_evidence": truncate_words(result.get("evidence", "")),
                    "deepseek_model": result.get("model", "") if candidate else "",
                    "prompt_version": (
                        "Chen-Sialm-Xu-2026-IA.4.1-original-mutual-fund-adaptation"
                        if candidate
                        else ""
                    ),
                    "api_input_sha256": candidate["input_sha256"] if candidate else "",
                    "api_error": result.get("error", "") if candidate else "",
                }
            )
    os.replace(temp, OUTPUT)


def main() -> None:
    """Screen the shards, classify the candidates from the cache or the API, settle
    identical inputs, and write the classification table and its usage report."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not LOG.exists():
        LOG.write_text("", encoding="utf-8")
    log("Reading canonical local fund-level shards; existing classifications are ignored")
    # The AI keyword screen on every usable fund-filing; each hit becomes a candidate with
    # its keyword-context excerpt.
    rows, source_stats = iter_shards()
    candidate_list = []
    screen_mismatches = 0
    for row in rows:
        recomputed = bool(AI_KEYWORD.search(row.get("strategy") or ""))
        screen_mismatches += int(recomputed != bool(row.get("ai_strategy")))
        if recomputed:
            candidate_list.append(candidate_record(row, args.model))
    write_candidates(candidate_list)
    # A profile of the inputs, written before any model call.
    by_year_funds = {
        str(year): len({row["series_id"] for row in rows if row["date"].startswith(str(year))})
        for year in range(2020, 2027)
    }
    by_year_candidates = {
        str(year): sum(candidate["filing_date"].startswith(str(year)) for candidate in candidate_list)
        for year in range(2020, 2027)
    }
    profile = {
        **source_stats,
        "latest_filing_date": max(row["date"] for row in rows),
        "candidate_fund_filings": len(candidate_list),
        "keyword_screen_mismatches_vs_stored_flag": screen_mismatches,
        "distinct_funds_by_year": by_year_funds,
        "candidate_rows_by_year": by_year_candidates,
        "candidate_input_file": tp.portable(CANDIDATES),
        "candidate_input_sha256": hashlib.sha256(CANDIDATES.read_bytes()).hexdigest(),
        "prompt_version": "Chen-Sialm-Xu-2026-IA.4.1-original-mutual-fund-adaptation",
        "model": args.model,
    }
    PROFILE.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    log(
        f"Usable fund-filings: {len(rows):,}; AI-keyword candidates: {len(candidate_list):,}; "
        f"latest date: {profile['latest_filing_date']}"
    )
    if args.prepare_only:
        log("Prepare-only mode complete; no API calls made")
        return

    # Classify the candidates that have no answer in the progress ledger yet.
    api_key = tp.api_key()
    if not api_key and not tp.NO_API:
        raise RuntimeError(f"API key file is empty: {API_KEY_FILE}")
    completed = load_progress()
    todo = [
        candidate
        for candidate in candidate_list
        if (candidate["accession"], candidate["series_id"], candidate["input_sha256"])
        not in completed
    ]
    if args.limit:
        todo = todo[: args.limit]
    log(f"DeepSeek model={args.model}; already complete={len(completed):,}; to call={len(todo):,}")
    tp.no_api_stop(len(todo), "AI-keyword candidates")

    started = time.time()
    errors = 0
    with PROGRESS.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:
        for index, result in enumerate(
            executor.map(lambda item: classify(item, api_key, args.model), todo), start=1
        ):
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            if result.get("error"):
                errors += 1
            if index % 50 == 0 or index == len(todo):
                elapsed = time.time() - started
                eta = (len(todo) - index) * elapsed / index if index else 0
                log(
                    f"API progress {index:,}/{len(todo):,}; errors={errors}; "
                    f"elapsed={elapsed / 60:.1f} min; ETA={eta / 60:.1f} min"
                )

    completed = load_progress()

    # Identical API inputs must not produce different labels merely because they
    # belong to two share classes or repeated filings.  Detect any such model
    # nondeterminism, adjudicate each conflicting text once, and apply that single
    # fresh answer to every row carrying the identical input hash.
    by_input_hash: dict[str, list[dict]] = collections.defaultdict(list)
    for candidate in candidate_list:
        by_input_hash[candidate["input_sha256"]].append(candidate)
    conflict_groups = []
    for input_sha, group in by_input_hash.items():
        decisions = {
            (
                completed[(item["accession"], item["series_id"], input_sha)].get("ai_category"),
                completed[(item["accession"], item["series_id"], input_sha)].get("is_quantitative"),
            )
            for item in group
            if (item["accession"], item["series_id"], input_sha) in completed
        }
        if len(decisions) > 1:
            conflict_groups.append(group)

    if conflict_groups:
        tp.no_api_stop(len(conflict_groups), "identical-input conflicts")
        log(
            f"Identical-input conflicts found: {len(conflict_groups)} groups; "
            "running one canonical adjudication per group"
        )
        adjudications = [classify(group[0], api_key, args.model) for group in conflict_groups]
        with PROGRESS.open("a", encoding="utf-8") as handle:
            for group, adjudication in zip(conflict_groups, adjudications):
                if adjudication.get("error"):
                    continue
                for item in group:
                    corrected = {
                        **adjudication,
                        "accession": item["accession"],
                        "series_id": item["series_id"],
                        "input_sha256": item["input_sha256"],
                        "adjudicated_identical_input": True,
                    }
                    handle.write(json.dumps(corrected, ensure_ascii=False) + "\n")
            handle.flush()
        completed = load_progress()

    # Every group of identical inputs must now carry a single answer.
    remaining_conflicts = 0
    for input_sha, group in by_input_hash.items():
        decisions = {
            (
                completed[(item["accession"], item["series_id"], input_sha)].get("ai_category"),
                completed[(item["accession"], item["series_id"], input_sha)].get("is_quantitative"),
            )
            for item in group
            if (item["accession"], item["series_id"], input_sha) in completed
        }
        remaining_conflicts += int(len(decisions) > 1)
    if remaining_conflicts:
        raise RuntimeError(f"{remaining_conflicts} identical-input conflict groups remain")

    # Write the full fund-filing table and the usage report.
    candidate_map = {(row["accession"], row["series_id"]): row for row in candidate_list}
    missing = [
        row
        for row in candidate_list
        if (row["accession"], row["series_id"], row["input_sha256"]) not in completed
    ]
    export_csv(rows, candidate_map, completed)
    call_counts = progress_call_counts()
    usage_report = {
        "prompt_tokens_this_run": _usage["prompt_tokens"],
        "completion_tokens_this_run": _usage["completion_tokens"],
        "api_calls_this_run": _usage["api_calls"],
        **call_counts,
        "total_successful_decision_calls_recorded": (
            call_counts["base_decision_calls"]
            + call_counts["identical_input_adjudication_calls"]
        ),
        "model": args.model,
        "candidates": len(candidate_list),
        "completed_for_current_inputs": len(candidate_list) - len(missing),
        "missing_or_failed_current_inputs": len(missing),
        "identical_input_conflict_groups_adjudicated_this_run": len(conflict_groups),
        "identical_input_conflict_groups_remaining": remaining_conflicts,
        "elapsed_seconds_this_run": round(time.time() - started, 2),
        "output_csv": tp.portable(OUTPUT),
        "output_csv_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
    }
    USAGE.write_text(json.dumps(usage_report, indent=2), encoding="utf-8")
    if missing:
        log(f"Incomplete: {len(missing)} candidate inputs still missing; run again to retry")
        raise RuntimeError(f"{len(missing)} candidate classifications are incomplete")
    log(
        f"Classification complete: {len(candidate_list):,}/{len(candidate_list):,} candidates; "
        f"final CSV rows={len(rows):,}"
    )


if __name__ == "__main__":
    main()

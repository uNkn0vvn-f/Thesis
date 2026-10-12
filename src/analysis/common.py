"""Shared data validation for research summaries."""
from __future__ import annotations

import pandas as pd


def require(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    """Raise if `frame` lacks any of `columns`; `name` identifies the table in the message."""
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")


def boolean(values: pd.Series) -> pd.Series:
    """Parse CSV booleans explicitly; missing flags follow the original False rule."""
    text = values.astype("string").str.strip().str.lower()
    allowed = {"true", "false", "1", "0", "1.0", "0.0", "t", "f", "yes", "no", ""}
    bad = text.notna() & ~text.isin(allowed)
    if bad.any():
        raise ValueError(f"Invalid boolean in {values.name}: {text[bad].iloc[0]!r}")
    return text.isin({"true", "1", "1.0", "t", "yes"}).fillna(False).astype(bool)

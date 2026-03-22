"""
Shared helpers for benchmark CSV I/O and latency/memory statistics.
"""

from __future__ import annotations

import csv
import math
import numbers
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

# Canonical schema written by ModelRunner (v2)
CSV_COLUMNS_V2: list[str] = [
    "run_id",
    "timestamp",
    "model_name",
    "backend",
    "num_parameters",
    "n_queries",
    "latency_mean_s",
    "latency_std_s",
    "latency_p95_s",
    "memory_mean_mb",
    "memory_std_mb",
    "tokens_per_sec_mean",
]

# Legacy columns (v1) — still readable by Visualizer
LEGACY_COLUMN_MAP = {
    "Model Name": "model_name",
    "Number of Model Parameters": "num_parameters",
    "Average Query Execution Time": "latency_mean_s",
    "Standard Deviation of Query Execution Time": "latency_std_s",
    "Average Memory Usage": "memory_mean_mb",
    "Standard Deviation of Memory Usage": "memory_std_mb",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def percentile(sorted_vals: Sequence[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])
    return float(sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f))


def latency_stats(seconds: Sequence[float]) -> dict[str, float]:
    vals = sorted(float(x) for x in seconds)
    n = len(vals)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "p95": float("nan")}
    mean = sum(vals) / n
    var = sum((x - mean) ** 2 for x in vals) / n
    std = var**0.5
    p95 = percentile(vals, 95.0)
    return {"mean": mean, "std": std, "p95": p95}


def memory_stats(mbs: Sequence[float]) -> dict[str, float]:
    vals = [float(x) for x in mbs]
    n = len(vals)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan")}
    mean = sum(vals) / n
    var = sum((x - mean) ** 2 for x in vals) / n
    return {"mean": mean, "std": var**0.5}


def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with canonical names; supports v1 legacy headers."""

    out = df.copy()
    rename = {}
    for old, new in LEGACY_COLUMN_MAP.items():
        if old in out.columns and new not in out.columns:
            rename[old] = new
    if rename:
        out = out.rename(columns=rename)
    return out


def migrate_llm_summary_if_needed(path: str) -> None:
    """If llmSummary.csv uses legacy v1 headers, rewrite in-place as v2 (preserves rows)."""
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        return
    with open(path, encoding="utf-8") as f:
        header = f.readline()
    if header.strip().startswith("run_id,"):
        return
    df = pd.read_csv(path)
    df = normalize_dataframe_columns(df)
    out = pd.DataFrame()
    for c in CSV_COLUMNS_V2:
        out[c] = df[c] if c in df.columns else np.nan
    out["run_id"] = out["run_id"].astype(object)
    mask = out["run_id"].isna()
    if mask.any():
        out.loc[mask, "run_id"] = [str(uuid.uuid4()) for _ in range(int(mask.sum()))]
    out["backend"] = out["backend"].fillna("legacy_import")
    out.to_csv(path, index=False)


def new_run_id() -> str:
    return str(uuid.uuid4())


def serialize_row_for_csv(row: Mapping[str, Any]) -> dict[str, Any]:
    """
    Convert row values to types the stdlib csv module always accepts (avoids numpy / torch
    scalars and NaN issues that can surface as TypeError or odd 'field' errors on write).
    """
    out: dict[str, Any] = {}
    for k in CSV_COLUMNS_V2:
        v = row[k]
        if v is None:
            out[k] = ""
            continue
        if isinstance(v, np.generic):
            v = v.item()
        elif hasattr(v, "item") and callable(v.item) and not isinstance(v, (bytes, str, bytearray)):
            try:
                v = v.item()
            except Exception:
                pass
        if isinstance(v, bool):
            out[k] = str(v).lower()
            continue
        if isinstance(v, numbers.Integral):
            out[k] = int(v)
            continue
        if isinstance(v, numbers.Real):
            fv = float(v)
            if math.isnan(fv) or math.isinf(fv):
                out[k] = ""
            else:
                out[k] = fv
            continue
        out[k] = str(v)
    return out


def append_result_row_v2(path: str, row: Mapping[str, Any]) -> None:
    """Append one v2 row; create file with header if missing."""
    migrate_llm_summary_if_needed(path)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    missing = [k for k in CSV_COLUMNS_V2 if k not in row]
    if missing:
        raise ValueError(f"Row missing keys: {missing}")
    safe = serialize_row_for_csv(row)
    write_header = not os.path.isfile(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS_V2, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow({k: safe[k] for k in CSV_COLUMNS_V2})

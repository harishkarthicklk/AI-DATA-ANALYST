from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .session import session
except ImportError:
    from session import session

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets"
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def safe_dataset_path(filename: str) -> Path:
    if not filename or not str(filename).strip():
        raise ValueError("Dataset filename cannot be empty.")
    candidate = Path(str(filename).strip().strip('"\''))
    if candidate.parts and candidate.parts[0].lower() == "datasets":
        candidate = Path(*candidate.parts[1:])
    if candidate.is_absolute():
        raise ValueError("Use a dataset filename inside the datasets folder.")
    path = (DATASET_DIR / candidate).resolve()
    root = DATASET_DIR.resolve()
    if root != path and root not in path.parents:
        raise ValueError("Dataset path must remain inside datasets folder.")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError("Supported datasets are CSV, XLSX and XLS files.")
    return path


def read_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    if df.empty:
        raise ValueError("The dataset contains no rows.")
    return df


def get_dataframe() -> pd.DataFrame:
    """Read the active dataset from the shared session using a safe path."""
    dataset = session.get_dataset()
    if not dataset:
        raise RuntimeError("No dataset is loaded. Use load_dataset first.")
    return read_dataset(safe_dataset_path(dataset))


def json_value(value: Any):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [{str(k): json_value(v) for k, v in row.items()} for row in df.to_dict(orient="records")]


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


# Generic aliases help map natural language to arbitrary column names.
ALIASES = {
    "sales": {"sales", "sale", "revenue", "income", "amount", "value", "turnover"},
    "amount": {"amount", "sales", "revenue", "income", "value", "price"},
    "quantity": {"quantity", "qty", "count", "units", "volume"},
    "date": {"date", "time", "timestamp", "day", "month", "year"},
    "region": {"region", "area", "zone", "territory", "location", "district"},
    "category": {"category", "type", "class", "group", "segment"},
    "customer": {"customer", "client", "buyer", "user", "account"},
    "representative": {"rep", "representative", "agent", "employee", "salesperson", "staff"},
    "latitude": {"latitude", "lat"},
    "longitude": {"longitude", "lon", "lng"},
}


def resolve_column(df: pd.DataFrame, phrase: str, *, numeric: bool | None = None) -> str:
    if not phrase:
        raise ValueError("A column description is required.")
    cols = [str(c) for c in df.columns]
    exact = {normalize(c): c for c in cols}
    p = normalize(phrase)
    if p in exact:
        return exact[p]

    query_tokens = set(p.split())
    best = None
    best_score = -1.0
    for col in cols:
        if numeric is True and not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if numeric is False and pd.api.types.is_numeric_dtype(df[col]):
            continue
        ct = set(normalize(col).split())
        score = 0.0
        score += 5.0 * len(query_tokens & ct)
        if p and p in normalize(col):
            score += 4.0
        for token in query_tokens:
            for alias_key, aliases in ALIASES.items():
                if token == alias_key or token in aliases:
                    if ct & aliases or alias_key in ct:
                        score += 2.5
        # Prefer numeric columns for amount-like words.
        if numeric is True and pd.api.types.is_numeric_dtype(df[col]):
            score += 0.5
        if score > best_score:
            best_score, best = score, col
    if best is None or best_score <= 0:
        raise ValueError(f"Could not map '{phrase}' to a dataset column. Available columns: {cols}")
    return best


def infer_numeric_column(df: pd.DataFrame, phrase: str) -> str:
    try:
        return resolve_column(df, phrase, numeric=True)
    except Exception:
        nums = list(df.select_dtypes(include="number").columns)
        if len(nums) == 1:
            return str(nums[0])
        raise


def summarize_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": [str(c) for c in df.columns],
        "data_types": {str(c): str(df[c].dtype) for c in df.columns},
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": [str(c) for c in df.select_dtypes(include="number").columns],
        "categorical_columns": [str(c) for c in df.select_dtypes(exclude="number").columns],
    }


def dataframe_to_json(data: Any) -> str:
    return json.dumps(data, indent=2, default=str)

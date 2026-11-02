from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from fastmcp import FastMCP

from core import get_dataframe, read_dataset, records, safe_dataset_path, summarize_dataframe
from session import session

mcp = FastMCP("AnalyticsAI Data Server")


@mcp.tool
def load_dataset(filename: str) -> dict[str, Any]:
    """Load a CSV or Excel dataset from datasets/."""
    try:
        path = safe_dataset_path(filename)
        df = read_dataset(path)
        # Store the dataset path relative to the project root so every MCP
        # server process can read the same session selection.
        relative = path.relative_to(path.parents[1]).as_posix()
        session.set_dataset(relative)
        return {"status": "success", "message": "Dataset loaded successfully!", "dataset": path.name,
                "path": str(path), "relative_path": relative, "rows": len(df), "columns": len(df.columns),
                "column_names": [str(c) for c in df.columns]}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@mcp.tool
def dataset_summary() -> dict[str, Any]:
    try:
        return {"status": "success", **summarize_dataframe(get_dataframe())}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@mcp.tool
def show_columns() -> dict[str, Any]:
    try:
        df = get_dataframe()
        return {"status": "success", "columns": [{"name": str(c), "dtype": str(df[c].dtype)} for c in df.columns]}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@mcp.tool
def head(n: int = 5) -> dict[str, Any]:
    try:
        n = max(1, min(int(n), 100))
        return {"status": "success", "rows": records(get_dataframe().head(n))}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@mcp.tool
def describe(column: str = "") -> dict[str, Any]:
    try:
        df = get_dataframe()
        if column and column not in df.columns:
            raise ValueError(f"Column '{column}' not found.")
        target = df[[column]] if column else df
        result = target.describe(include="all").transpose().reset_index(names="column")
        result = result.replace([np.inf, -np.inf], np.nan)
        return {"status": "success", "statistics": records(result)}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@mcp.tool
def missing_values() -> dict[str, Any]:
    try:
        df = get_dataframe()
        result = df.isna().sum().reset_index()
        result.columns = ["column", "missing_values"]
        result["missing_percent"] = (result["missing_values"] / len(df) * 100).round(2)
        return {"status": "success", "missing_values": records(result)}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@mcp.tool
def duplicate_count() -> dict[str, Any]:
    try:
        return {"status": "success", "duplicate_rows": int(get_dataframe().duplicated().sum())}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@mcp.tool
def unique_values(column: str) -> dict[str, Any]:
    try:
        df = get_dataframe()
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found.")
        vals = df[column].dropna().unique().tolist()
        return {"status": "success", "column": column, "unique_count": int(df[column].nunique(dropna=True)),
                "values": [str(v) if not isinstance(v, (int, float, bool)) else v for v in vals[:500]],
                "truncated": len(vals) > 500}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@mcp.tool
def value_counts(column: str, limit: int = 20) -> dict[str, Any]:
    try:
        df = get_dataframe()
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found.")
        limit = max(1, min(int(limit), 100))
        result = df[column].value_counts(dropna=False).head(limit).reset_index()
        result.columns = [column, "count"]
        return {"status": "success", "column": column, "counts": records(result)}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@mcp.tool
def group_by_aggregate(group_column: str, value_column: str, aggregation: str = "sum") -> dict[str, Any]:
    """Group by one column and calculate sum, mean, count, min, max or median."""
    try:
        df = get_dataframe()
        if group_column not in df.columns or value_column not in df.columns:
            raise ValueError(f"Columns must exist. Available columns: {[str(c) for c in df.columns]}")
        aggregation = aggregation.lower().strip()
        allowed = {"sum", "mean", "count", "min", "max", "median"}
        if aggregation not in allowed:
            raise ValueError(f"Unsupported aggregation: {aggregation}")
        if aggregation == "count":
            result = df.groupby(group_column, dropna=False)[value_column].count().reset_index(name=value_column)
        else:
            if not pd.api.types.is_numeric_dtype(df[value_column]):
                raise ValueError(f"Column '{value_column}' must be numeric for {aggregation}.")
            result = df.groupby(group_column, dropna=False)[value_column].agg(aggregation).reset_index()
        result = result.sort_values(value_column, ascending=False, na_position="last")
        return {"status": "success", "group_column": group_column, "value_column": value_column,
                "aggregation": aggregation, "result": records(result)}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@mcp.tool
def filter_rows(column: str, operator: str, value: str, limit: int = 5) -> dict[str, Any]:
    try:
        df = get_dataframe()
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found.")
        operator = operator.strip().lower()
        if operator not in {"=", "==", "!=", ">", ">=", "<", "<=", "contains"}:
            raise ValueError("Unsupported operator.")
        series = df[column]
        compare: Any = value
        if pd.api.types.is_numeric_dtype(series):
            compare = float(value)
        if operator in {"=", "=="}: mask = series == compare
        elif operator == "!=": mask = series != compare
        elif operator == ">": mask = series > compare
        elif operator == ">=": mask = series >= compare
        elif operator == "<": mask = series < compare
        elif operator == "<=": mask = series <= compare
        else: mask = series.astype(str).str.contains(str(value), case=False, na=False)
        limit = max(1, min(int(limit), 5))
        return {"status": "success", "matched_rows": int(mask.sum()), "rows": records(df.loc[mask].head(limit))}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@mcp.tool
def sort_rows(column: str, descending: bool = True, limit: int = 5) -> dict[str, Any]:
    try:
        df = get_dataframe()
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found.")
        limit = max(1, min(int(limit), 5))
        return {"status": "success", "column": column, "descending": bool(descending),
                "rows": records(df.sort_values(column, ascending=not descending, na_position="last").head(limit))}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@mcp.tool
def correlation_matrix() -> dict[str, Any]:
    try:
        numeric = get_dataframe().select_dtypes(include="number")
        if len(numeric.columns) < 2:
            return {"status": "success", "message": "At least two numeric columns are required.", "correlation": {}}
        return {"status": "success", "correlation": numeric.corr().round(4).to_dict()}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


if __name__ == "__main__":
    mcp.run()

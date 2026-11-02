from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from fastmcp import FastMCP

from core import get_dataframe, resolve_column

mcp = FastMCP("AnalyticsAI Visualization Server")
BASE_DIR = Path(__file__).resolve().parent.parent
CHART_DIR = BASE_DIR / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED = {"none", "sum", "mean", "count", "min", "max", "median"}


def _path(kind: str) -> Path:
    return CHART_DIR / f"{kind}_{datetime.now():%Y%m%d_%H%M%S_%f}.png"


def _aggregate(df, x, y, aggregation):
    if aggregation not in ALLOWED:
        raise ValueError(f"aggregation must be one of {sorted(ALLOWED)}")
    if aggregation == "none":
        out = df[[x, y]].copy()
    elif aggregation == "count":
        out = df.groupby(x, dropna=False)[y].count().reset_index(name=y)
    else:
        if not pd.api.types.is_numeric_dtype(df[y]):
            raise ValueError(f"'{y}' must be numeric for {aggregation}.")
        out = df.groupby(x, dropna=False)[y].agg(aggregation).reset_index()
    return out.dropna(subset=[x, y])


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path)


@mcp.tool
def bar_chart(x_column: str, y_column: str, aggregation: str = "sum") -> dict:
    """Create a bar chart. Example: Region + Sales_Amount + sum."""
    try:
        df = _aggregate(get_dataframe(), x_column, y_column, aggregation)
        if df.empty: raise ValueError("No plottable rows remain.")
        if aggregation != "none": df = df.sort_values(y_column, ascending=False).head(30)
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.bar(df[x_column].astype(str), df[y_column])
        ax.set_xlabel(x_column); ax.set_ylabel(y_column); ax.set_title(f"{aggregation.title()} of {y_column} by {x_column}")
        ax.tick_params(axis="x", rotation=35)
        return {"status":"success","chart_type":"bar","path":_save(fig,_path("bar_chart")),"rows_plotted":len(df)}
    except Exception as e: return {"status":"error","message":str(e)}


@mcp.tool
def line_chart(x_column: str, y_column: str, aggregation: str = "mean") -> dict:
    """Create a line chart; use a date/time column for x when appropriate."""
    try:
        df = get_dataframe().copy()
        if x_column not in df.columns or y_column not in df.columns: raise ValueError("Requested columns do not exist.")
        if aggregation != "none": df = _aggregate(df, x_column, y_column, aggregation)
        else: df = df[[x_column,y_column]].dropna()
        if df.empty: raise ValueError("No plottable rows remain.")
        if not pd.api.types.is_numeric_dtype(df[y_column]): raise ValueError(f"'{y_column}' must be numeric.")
        if pd.api.types.is_datetime64_any_dtype(df[x_column]): df=df.sort_values(x_column)
        fig, ax=plt.subplots(figsize=(10,5.5)); ax.plot(df[x_column].astype(str),df[y_column],marker="o",markersize=3)
        ax.set_xlabel(x_column); ax.set_ylabel(y_column); ax.set_title(f"{aggregation.title()} of {y_column} by {x_column}"); ax.tick_params(axis="x",rotation=35)
        return {"status":"success","chart_type":"line","path":_save(fig,_path("line_chart")),"rows_plotted":len(df)}
    except Exception as e: return {"status":"error","message":str(e)}


@mcp.tool
def pie_chart(category_column: str, value_column: str, aggregation: str = "sum") -> dict:
    """Create a pie chart of a grouped numeric value or count."""
    try:
        df=_aggregate(get_dataframe(),category_column,value_column,aggregation).sort_values(value_column,ascending=False).head(10)
        if df.empty: raise ValueError("No plottable rows remain.")
        vals=df[value_column]
        if (vals < 0).any(): raise ValueError("Pie charts require non-negative values.")
        fig,ax=plt.subplots(figsize=(7,7)); ax.pie(vals,labels=df[category_column].astype(str),autopct="%1.1f%%"); ax.set_title(f"{aggregation.title()} of {value_column} by {category_column}")
        return {"status":"success","chart_type":"pie","path":_save(fig,_path("pie_chart")),"categories":len(df)}
    except Exception as e:return {"status":"error","message":str(e)}


@mcp.tool
def histogram(column: str, bins: int = 20) -> dict:
    try:
        df=get_dataframe();
        if column not in df.columns: raise ValueError(f"Column '{column}' not found.")
        if not pd.api.types.is_numeric_dtype(df[column]): raise ValueError("Histogram column must be numeric.")
        fig,ax=plt.subplots(figsize=(9,5)); ax.hist(df[column].dropna(),bins=max(5,min(int(bins),100))); ax.set_xlabel(column); ax.set_ylabel("Frequency"); ax.set_title(f"Distribution of {column}")
        return {"status":"success","chart_type":"histogram","path":_save(fig,_path("histogram"))}
    except Exception as e:return {"status":"error","message":str(e)}


@mcp.tool
def scatter_plot(x_column: str, y_column: str) -> dict:
    try:
        df=get_dataframe();
        if not pd.api.types.is_numeric_dtype(df[x_column]) or not pd.api.types.is_numeric_dtype(df[y_column]): raise ValueError("Scatter plot requires two numeric columns.")
        df=df[[x_column,y_column]].dropna(); fig,ax=plt.subplots(figsize=(8,5)); ax.scatter(df[x_column],df[y_column],alpha=.65); ax.set_xlabel(x_column); ax.set_ylabel(y_column); ax.set_title(f"{y_column} vs {x_column}")
        return {"status":"success","chart_type":"scatter","path":_save(fig,_path("scatter_plot")),"rows_plotted":len(df)}
    except Exception as e:return {"status":"error","message":str(e)}


@mcp.tool
def box_plot(column: str) -> dict:
    try:
        df=get_dataframe();
        if not pd.api.types.is_numeric_dtype(df[column]): raise ValueError("Box plot requires a numeric column.")
        fig,ax=plt.subplots(figsize=(7,5)); ax.boxplot(df[column].dropna()); ax.set_ylabel(column); ax.set_title(f"Box Plot of {column}")
        return {"status":"success","chart_type":"box","path":_save(fig,_path("box_plot"))}
    except Exception as e:return {"status":"error","message":str(e)}


@mcp.tool
def correlation_heatmap() -> dict:
    try:
        df=get_dataframe().select_dtypes(include="number")
        if len(df.columns)<2: raise ValueError("At least two numeric columns are required.")
        fig,ax=plt.subplots(figsize=(9,7)); sns.heatmap(df.corr(),annot=True,fmt=".2f",ax=ax); ax.set_title("Correlation Heatmap")
        return {"status":"success","chart_type":"heatmap","path":_save(fig,_path("heatmap"))}
    except Exception as e:return {"status":"error","message":str(e)}


@mcp.tool
def list_saved_charts() -> dict:
    return {"status":"success","charts":[str(p) for p in sorted(CHART_DIR.glob("*.png"), key=lambda x:x.stat().st_mtime, reverse=True)[:100]]}


if __name__ == "__main__": mcp.run()

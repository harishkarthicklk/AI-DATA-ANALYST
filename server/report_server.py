from __future__ import annotations

import re
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from fastmcp import FastMCP
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from core import get_dataframe
from session import session

mcp = FastMCP("AnalyticsAI Report Server")
BASE = Path(__file__).resolve().parent.parent
REPORTS = BASE / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)


def safe_text(value: Any) -> str:
    return escape(str(value))


def make_table(rows: list[list[Any]]) -> Table:
    converted = [[safe_text(value) for value in row] for row in rows]
    t = Table(converted, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#203864")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), .3, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F5FA")]),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def choose_columns(df: pd.DataFrame, request: str) -> tuple[str | None, str | None]:
    """Choose relevant x/y columns from the request, falling back sensibly."""
    low = request.lower()
    columns = [str(c) for c in df.columns]
    numeric = [str(c) for c in df.select_dtypes(include="number").columns]
    categorical = [str(c) for c in df.select_dtypes(exclude="number").columns]

    def mentioned(c: str) -> bool:
        return re.search(r"(?<![a-z0-9_])" + re.escape(c.lower()) + r"(?![a-z0-9_])", low) is not None

    mentioned_numeric = [c for c in numeric if mentioned(c)]
    y = mentioned_numeric[0] if mentioned_numeric else None
    x = next((c for c in categorical if mentioned(c)), None)

    # Relationship/correlation requests should compare the two requested
    # numeric columns, not accidentally group by a high-cardinality ID field.
    if any(word in low for word in ("correlation", "relationship", "versus", " vs ", "against")) and len(mentioned_numeric) >= 2:
        return mentioned_numeric[0], mentioned_numeric[1]

    # For spatial requests, use the requested geographic coordinates where relevant.
    if any(w in low for w in ("spatial", "location", "geographic", "latitude", "longitude", "map")):
        if "longitude" in columns and "latitude" in columns:
            x, y = "longitude", "latitude"
    if x is None and categorical:
        # Prefer meaningful group labels over row identifiers such as IDs/keys.
        groupable = [c for c in categorical if not re.search(r"(^|_)(id|key|code|uuid)$", c.lower()) and not c.lower().endswith(("_id", "_key", "_code"))]
        x = groupable[0] if groupable else categorical[0]
    if y is None and numeric:
        # Pick the first numeric field not already serving as x.
        y = next((c for c in numeric if c != x), numeric[0])
    return x, y


def safe_filename_part(value: str, limit: int = 44) -> str:
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return (value or "dataset_report")[:limit]


@mcp.tool
def export_pdf_report(
    title: str = "AnalyticsAI Dataset Report",
    user_request: str = "",
) -> dict[str, Any]:
    """Create a fresh PDF report tailored to the user's request and active dataset.

    Args:
        title: Report title, preferably the user's requested report description.
        user_request: Original natural-language request; it controls the report focus.
    """
    chart_paths: list[Path] = []
    try:
        df = get_dataframe()
        if df.empty:
            raise ValueError("The selected dataset contains no rows.")

        request = (user_request or title or "dataset overview").strip()
        low = request.lower()
        styles = getSampleStyleSheet()
        now = datetime.now()
        stamp = now.strftime("%Y%m%d_%H%M%S_%f")
        pdf = REPORTS / f"{safe_filename_part(title)}_{stamp}.pdf"
        doc = SimpleDocTemplate(
            str(pdf), pagesize=landscape(A4), rightMargin=28,
            leftMargin=28, topMargin=28, bottomMargin=28,
        )
        dataset = session.get_dataset() or "Unknown dataset"
        numeric = list(df.select_dtypes(include="number").columns)
        categorical = list(df.select_dtypes(exclude="number").columns)
        elements = [
            Paragraph(safe_text(title or "AnalyticsAI Dataset Report"), styles["Title"]),
            Spacer(1, 8),
            Paragraph(f"Dataset: {safe_text(dataset)}", styles["Normal"]),
            Paragraph(f"Generated: {now:%d-%m-%Y %H:%M:%S}", styles["Normal"]),
            Paragraph(f"Report focus: {safe_text(request)}", styles["Normal"]),
            Spacer(1, 14),
        ]

        # Overview always reflects the live selected dataset.
        overview = [
            ["Metric", "Value"],
            ["Rows", len(df)], ["Columns", len(df.columns)],
            ["Missing cells", int(df.isna().sum().sum())],
            ["Duplicate rows", int(df.duplicated().sum())],
            ["Numeric columns", len(numeric)],
            ["Categorical columns", len(categorical)],
        ]
        elements += [Paragraph("Dataset overview", styles["Heading1"]), make_table(overview), Spacer(1, 12)]

        elements += [Paragraph("Column profile", styles["Heading1"]), make_table(
            [["Column", "Data type", "Missing", "Unique values"]] + [
                [str(c), str(df[c].dtype), int(df[c].isna().sum()), int(df[c].nunique(dropna=True))]
                for c in df.columns
            ]
        ), Spacer(1, 12)]

        # Request-specific content: include only sections relevant to the prompt,
        # while still retaining a useful profile for generic report requests.
        wants_stats = any(w in low for w in ("statistic", "describe", "distribution", "summary", "overview", "report", "pdf", "numeric", "correlation"))
        if numeric and wants_stats:
            desc = df[numeric].describe().round(3).transpose().reset_index()
            desc = desc.rename(columns={"index": "Column"})
            elements += [Paragraph("Descriptive statistics", styles["Heading1"]), make_table(
                [list(desc.columns)] + [[str(v) for v in row] for row in desc.itertuples(index=False, name=None)]
            ), Spacer(1, 12)]

        if "missing" in low or "null" in low or "quality" in low or "clean" in low:
            missing = df.isna().sum()
            elements += [Paragraph("Missing-value analysis", styles["Heading1"]), make_table(
                [["Column", "Missing count", "Missing %"]] + [
                    [str(c), int(missing[c]), round(float(missing[c]) / len(df) * 100, 2)]
                    for c in df.columns
                ]
            ), Spacer(1, 12)]

        if "correlation" in low and len(numeric) >= 2:
            corr = df[numeric].corr(numeric_only=True).round(3)
            elements += [Paragraph("Correlation matrix", styles["Heading1"]), make_table(
                [["Column"] + list(map(str, corr.columns))] + [
                    [str(index)] + [str(v) for v in row]
                    for index, row in zip(corr.index, corr.to_numpy())
                ]
            ), Spacer(1, 12)]

        x_col, y_col = choose_columns(df, request)
        wants_chart = any(w in low for w in ("chart", "plot", "graph", "distribution", "trend", "compare", "comparison", "visual", "spatial", "location", "geographic", "latitude", "longitude", "report", "pdf", "analysis"))
        if wants_chart and y_col and pd.api.types.is_numeric_dtype(df[y_col]):
            # Category-vs-measure requests get a fresh grouped bar chart.
            if x_col and x_col != y_col and not pd.api.types.is_numeric_dtype(df[x_col]):
                grouped = df.groupby(x_col, dropna=False)[y_col].mean().sort_values(ascending=False).head(15)
                if len(grouped):
                    image_path = REPORTS / f"report_bar_{stamp}.png"
                    fig, ax = plt.subplots(figsize=(9, 4.5))
                    ax.bar(grouped.index.astype(str), grouped.values)
                    ax.set_title(f"Average {y_col} by {x_col}")
                    ax.set_ylabel(f"Average {y_col}")
                    ax.tick_params(axis="x", rotation=35)
                    fig.tight_layout()
                    fig.savefig(image_path, dpi=140, bbox_inches="tight")
                    plt.close(fig)
                    chart_paths.append(image_path)
                    elements += [Paragraph(f"Average {safe_text(y_col)} by {safe_text(x_col)}", styles["Heading2"]), Image(str(image_path), width=7*inch, height=3.4*inch), Spacer(1, 8)]
            elif x_col and y_col and x_col in numeric and x_col != y_col:
                clean = df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
                if len(clean):
                    image_path = REPORTS / f"report_scatter_{stamp}.png"
                    fig, ax = plt.subplots(figsize=(9, 4.5))
                    ax.scatter(clean[x_col], clean[y_col], alpha=.55, s=14)
                    ax.set_xlabel(x_col); ax.set_ylabel(y_col)
                    ax.set_title(f"{y_col} versus {x_col}")
                    fig.tight_layout(); fig.savefig(image_path, dpi=140, bbox_inches="tight"); plt.close(fig)
                    chart_paths.append(image_path)
                    elements += [Paragraph(f"{safe_text(y_col)} versus {safe_text(x_col)}", styles["Heading2"]), Image(str(image_path), width=7*inch, height=3.4*inch)]
            else:
                values = pd.to_numeric(df[y_col], errors="coerce").dropna()
                if len(values):
                    image_path = REPORTS / f"report_hist_{stamp}.png"
                    fig, ax = plt.subplots(figsize=(9, 4.5))
                    ax.hist(values, bins=20)
                    ax.set_title(f"Distribution of {y_col}"); ax.set_xlabel(y_col); ax.set_ylabel("Frequency")
                    fig.tight_layout(); fig.savefig(image_path, dpi=140, bbox_inches="tight"); plt.close(fig)
                    chart_paths.append(image_path)
                    elements += [Paragraph(f"Distribution of {safe_text(y_col)}", styles["Heading2"]), Image(str(image_path), width=7*inch, height=3.4*inch)]

        # Always include a small live data sample, which makes reports useful
        # and ensures the content reflects the selected data rather than a template.
        sample = df.head(5).copy().fillna("")
        elements += [Spacer(1, 10), Paragraph("Sample records", styles["Heading1"]), make_table(
            [[str(c) for c in sample.columns]] + [[str(v)[:100] for v in row] for row in sample.astype(str).itertuples(index=False, name=None)]
        )]
        doc.build(elements)
        return {
            "status": "success",
            "message": "A new request-specific report was generated from the currently selected dataset.",
            "path": str(pdf),
            "dataset": str(dataset),
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "request": request,
            "chart_paths": [str(path) for path in chart_paths],
        }
    except Exception as exc:
        return {"status": "error", "error_type": type(exc).__name__, "message": str(exc)}
    finally:
        plt.close("all")


if __name__ == "__main__":
    mcp.run()

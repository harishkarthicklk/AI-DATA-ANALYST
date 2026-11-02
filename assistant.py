
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from rich.console import Console
from rich.panel import Panel

load_dotenv()

console = Console()
BASE_DIR = Path(__file__).resolve().parent


# --------------------------------------------------
# MCP SERVER CONFIGURATION
# --------------------------------------------------

def server(script):
    return {
        "transport": "stdio",
        "command": sys.executable,
        "args": [str(BASE_DIR / "server" / script)],
    }


def canonical_tool_name(name: str) -> str:
    """Normalize adapter-generated server_tool names to their local tool name."""
    # langchain-mcp-adapters commonly prefixes tool names with the server key.
    value = str(name)
    for prefix in ("data_analyst_", "visualization_", "report_", "filesystem_", "sqlite_", "spatial_sequential_"):
        if value.startswith(prefix):
            return value[len(prefix):]
    return value


def tool_map(tools):
    mapped = {}
    for tool in tools:
        mapped[tool.name] = tool
        short = canonical_tool_name(tool.name)
        # Keep first registration for duplicate generic names; server-prefixed
        # names remain available for unambiguous access.
        mapped.setdefault(short, tool)
    return mapped


def find_tool(tools, name):
    return next((tool for tool in tools if tool.name == name or canonical_tool_name(tool.name) == name), None)


def words(text):
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def parse_load(text):
    match = re.search(
        r"(?:load|open|use|import)\s+(?:the\s+)?"
        r"(?:dataset\s+)?[\"']?([^\"'\s]+\.(?:csv|xlsx|xls))[\"']?",
        text,
        re.IGNORECASE,
    )
    return match.group(1) if match else None


def extract_number(text, default=10):
    match = re.search(r"\b(\d+)\b", text)
    return int(match.group(1)) if match else default


# --------------------------------------------------
# TASK ROUTING
# --------------------------------------------------

def choose_group(text):
    w = words(text)

    if w & {"sql", "sqlite", "query", "database", "table"}:
        return "sql"

    if w & {
        "dbscan", "gwr", "kriging", "spatial", "geographic",
        "latitude", "longitude", "coordinate", "colocation",
        "outlier", "interpolation",
    }:
        return "spatial"

    if w & {
        "prefixspan", "gsp", "hmm", "dtw", "trajectory",
        "sequence", "sequential", "state-transition",
    }:
        return "sequential"

    if w & {
        "chart", "plot", "graph", "histogram", "scatter",
        "boxplot", "heatmap", "visualize", "visualization",
    }:
        return "visual"

    if w & {"report", "pdf"}:
        return "report"

    if w & {"file", "files", "folder", "read", "write", "search"}:
        return "filesystem"

    return "data"


# --------------------------------------------------
# LOCAL LLM
# --------------------------------------------------

def make_llm():
    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "qwen3:8b"),
        base_url=os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434",
        ),
        temperature=0,
        num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "4096")),
        num_predict=int(os.getenv("OLLAMA_NUM_PREDICT", "512")),
    )


# --------------------------------------------------
# AGENT INSTRUCTIONS
# --------------------------------------------------

SYSTEMS = {
    "data": (
        "You are a local data analyst. Use the supplied MCP tools. "
        "Inspect the schema when column mapping is uncertain. "
        "Never invent columns or results. Execute the user's request "
        "and report actual results. Keep answers concise."
    ),
    "visual": (
        "You create data visualizations. Use the supplied visualization "
        "MCP tools. Inspect the dataset columns when needed. Use the "
        "correct tool parameters and report the actual result and file path."
    ),
    "sql": (
        "You are a read-only SQLite analyst. The current dataset is "
        "synchronized to a table named data. Use only SELECT or WITH "
        "queries. Execute the user's request and report actual results."
    ),
    "spatial": (
        "You are a spatial data analyst. Use the supplied spatial MCP tools. "
        "Inspect the dataset schema before selecting columns. Never invent "
        "coordinates, values, or results. Use actual tool outputs."
    ),
    "sequential": (
        "You are a sequential data-mining analyst. Use the supplied "
        "sequential MCP tools. Inspect the schema when needed and explain "
        "the actual result briefly."
    ),
    "report": (
        "Generate dataset PDF reports using the supplied export_pdf_report "
        "tool. Report the actual output path."
    ),
    "filesystem": (
        "You manage files inside the project. Use the filesystem MCP tools. "
        "Never access paths outside the allowed project folders."
    ),
}


GROUP_TOOLS = {
    "data": [
        "dataset_summary", "show_columns", "head", "describe",
        "missing_values", "duplicate_count", "unique_values",
        "value_counts", "group_by_aggregate", "filter_rows",
        "sort_rows", "correlation_matrix",
    ],
    "visual": [
        "show_columns", "bar_chart", "line_chart", "pie_chart",
        "histogram", "scatter_plot", "box_plot",
        "correlation_heatmap", "list_saved_charts",
    ],
    "sql": [
        "list_tables", "show_schema", "count_rows",
        "show_first_rows", "execute_query", "sync_current_dataset",
    ],
    "spatial": [
        "show_columns", "run_dbscan", "run_gwr",
        "run_spatial_colocation", "run_kriging_interpolation",
        "run_spatial_outlier_detection",
    ],
    "sequential": [
        "show_columns", "run_prefixspan", "run_gsp",
        "run_hmm", "run_dtw", "run_trajectory_clustering",
    ],
    "report": ["export_pdf_report"],
    "filesystem": [
        "list_files", "read_file", "search_files",
        "write_file", "file_info",
    ],
}


# --------------------------------------------------
# COLUMN MATCHING
# --------------------------------------------------

def resolve_column_locally(
    df_columns,
    phrase,
    numeric_columns=None,
):
    columns = list(map(str, df_columns))
    phrase_clean = re.sub(
        r"[^a-z0-9]+",
        " ",
        phrase.lower(),
    ).strip()

    phrase_tokens = set(phrase_clean.split())

    aliases = {
        "sales": {
            "sales", "sale", "revenue", "amount", "income", "turnover",
        },
        "quantity": {
            "quantity", "qty", "units", "volume",
        },
        "region": {
            "region", "area", "zone", "territory", "location",
        },
        "category": {
            "category", "type", "class", "group", "segment",
        },
        "customer": {
            "customer", "client", "buyer", "user",
        },
        "date": {
            "date", "time", "timestamp", "month", "year",
        },
    }

    best_column = None
    best_score = -1

    for column in columns:
        column_tokens = set(
            re.sub(
                r"[^a-z0-9]+",
                " ",
                column.lower(),
            ).split()
        )

        score = 5 * len(phrase_tokens & column_tokens)

        if phrase_clean and phrase_clean in column.lower():
            score += 4

        for token in phrase_tokens:
            for key, alias_set in aliases.items():
                if token == key or token in alias_set:
                    if column_tokens & alias_set or key in column_tokens:
                        score += 2.5

        if (
            numeric_columns is not None
            and column in numeric_columns
            and phrase_tokens
            & {"sales", "revenue", "amount", "quantity", "value", "income"}
        ):
            score += 2

        if score > best_score:
            best_score = score
            best_column = column

    if best_column is None or best_score <= 0:
        raise ValueError(
            f"Could not map '{phrase}' to a dataset column."
        )

    return best_column


# --------------------------------------------------
# MCP TOOL EXECUTION
# --------------------------------------------------

def normalize_tool_result(result):
    """Unwrap MCP content blocks into normal Python/JSON values.

    The MCP adapter may return a list of text content blocks instead of a
    dictionary. Normalize that shape once so deterministic routes and schema
    inspection can consistently check status/columns.
    """
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        try:
            return json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return result
    if isinstance(result, list):
        parsed_items = []
        text_items = []
        for item in result:
            if isinstance(item, dict):
                content_text = item.get("text")
                item_type = item.get("type")
            else:
                content_text = getattr(item, "text", None)
                item_type = getattr(item, "type", None)
            if isinstance(content_text, str):
                text_items.append(content_text)
                try:
                    parsed_items.append(json.loads(content_text))
                except (json.JSONDecodeError, TypeError):
                    pass
            elif isinstance(item, dict) and item_type not in {"image", "resource"}:
                parsed_items.append(item)
        if len(parsed_items) == 1:
            return parsed_items[0]
        if parsed_items:
            return parsed_items
        if text_items:
            return "\n".join(text_items)
    # Some MCP clients expose CallToolResult-like objects with .content.
    content = getattr(result, "content", None)
    if isinstance(content, list):
        return normalize_tool_result(content)
    return result


async def call_tool(tool, args):
    if tool is None:
        return {
            "status": "error",
            "message": "The required MCP tool is not available.",
        }

    raw_result = await asyncio.wait_for(
        tool.ainvoke(args),
        timeout=float(os.getenv("MCP_TOOL_TIMEOUT", "30")),
    )
    return normalize_tool_result(raw_result)


async def get_schema(tool_map_dict):
    tool = tool_map_dict.get("show_columns")

    if tool is None:
        return None

    result = await call_tool(tool, {})

    if not isinstance(result, dict):
        return None

    if result.get("status") != "success":
        return None

    return result


# --------------------------------------------------
# SPATIAL ANALYSIS ROUTING
# --------------------------------------------------

async def spatial_route(text, tm):
    low = text.lower()

    schema = await get_schema(tm)

    if schema is None:
        return {
            "status": "error",
            "message": (
                "Could not inspect the dataset. "
                "Load a dataset first and check the MCP server."
            ),
        }

    columns_info = schema.get("columns", [])
    columns = {
        item["name"] for item in columns_info
        if isinstance(item, dict) and "name" in item
    }

    numeric_columns = {
        item["name"]
        for item in columns_info
        if isinstance(item, dict)
        and "name" in item
        and any(
            dtype in str(item.get("dtype", "")).lower()
            for dtype in ("int", "float", "double")
        )
    }

    if not {"latitude", "longitude"}.issubset(columns):
        return {
            "status": "error",
            "message": (
                "This spatial operation requires latitude and longitude "
                "columns. Available columns: "
                + ", ".join(sorted(columns))
            ),
        }

    # DBSCAN clustering
    if "dbscan" in low or "density clustering" in low:
        eps_match = re.search(
            r"(?:eps|distance|radius)\s*(?:=|of|is)?\s*(\d+(?:\.\d+)?)",
            low,
        )
        min_match = re.search(
            r"(?:min_samples|minimum samples)\s*(?:=|of|is)?\s*(\d+)",
            low,
        )

        eps_km = float(eps_match.group(1)) if eps_match else 1.0
        min_samples = int(min_match.group(1)) if min_match else 5

        return await call_tool(
            tm.get("run_dbscan"),
            {
                "latitude_col": "latitude",
                "longitude_col": "longitude",
                "eps_km": eps_km,
                "min_samples": min_samples,
            },
        )

    # GWR
    if "gwr" in low or "geographically weighted regression" in low:
        target = next(
            (
                column
                for column in ["house_value", "air_pollution_index"]
                if column in numeric_columns
            ),
            None,
        )

        if target is None:
            return {
                "status": "error",
                "message": (
                    "GWR needs a numeric target column. "
                    "Available numeric columns: "
                    + ", ".join(sorted(numeric_columns))
                ),
            }

        return await call_tool(
            tm.get("run_gwr"),
            {
                "latitude_col": "latitude",
                "longitude_col": "longitude",
                "target_col": target,
                "bandwidth_km": 5.0,
            },
        )

    # Spatial outlier detection
    if "spatial outlier" in low or "outlier detection" in low:
        attribute = next(
            (
                column
                for column in ["air_pollution_index", "house_value"]
                if column in numeric_columns
            ),
            None,
        )

        if attribute is None:
            return {
                "status": "error",
                "message": "No suitable numeric attribute was found.",
            }

        return await call_tool(
            tm.get("run_spatial_outlier_detection"),
            {
                "latitude_col": "latitude",
                "longitude_col": "longitude",
                "attribute_col": attribute,
                "k_neighbors": 5,
                "threshold_z": 2.5,
            },
        )

    # Spatial co-location
    if "colocation" in low or "co-location" in low:
        category = next(
            (
                column
                for column in ["store_category", "category", "type"]
                if column in columns
            ),
            None,
        )

        if category is None:
            return {
                "status": "error",
                "message": (
                    "No suitable category column was found. "
                    "Available columns: "
                    + ", ".join(sorted(columns))
                ),
            }

        return await call_tool(
            tm.get("run_spatial_colocation"),
            {
                "latitude_col": "latitude",
                "longitude_col": "longitude",
                "category_col": category,
                "distance_threshold_km": 1.0,
            },
        )

    # Kriging / interpolation: require target coordinates from the user.
    if "kriging" in low or "interpolation" in low:
        coordinate_match = re.search(
            r"(?:at|for|target)\s*"
            r"\(?\s*(-?\d+(?:\.\d+)?)\s*[, ]\s*"
            r"(-?\d+(?:\.\d+)?)\s*\)?",
            low,
        )

        if coordinate_match is None:
            return {
                "status": "needs_input",
                "message": (
                    "Please provide the target latitude and longitude. "
                    "Example: Interpolate house_value at 37.78, -122.40."
                ),
            }

        target_lat = float(coordinate_match.group(1))
        target_lon = float(coordinate_match.group(2))

        value_column = next(
            (
                column
                for column in ["house_value", "air_pollution_index"]
                if column in numeric_columns
            ),
            None,
        )

        if value_column is None:
            return {
                "status": "error",
                "message": "No suitable numeric column was found to interpolate.",
            }

        return await call_tool(
            tm.get("run_kriging_interpolation"),
            {
                "value_col": value_column,
                "latitude_col": "latitude",
                "longitude_col": "longitude",
                "target_lat": target_lat,
                "target_lon": target_lon,
            },
        )

    return None


# --------------------------------------------------
# SEQUENTIAL ANALYSIS ROUTING
# --------------------------------------------------

async def sequential_route(text, tm):
    low = text.lower()
    schema = await get_schema(tm)
    if not schema:
        return {"status": "error", "message": "Could not inspect the active dataset. Load a dataset first."}
    info = schema.get("columns", [])
    columns = [item.get("name") for item in info if isinstance(item, dict) and item.get("name")]
    # Select columns by common names in the supplied data; ask rather than
    # inventing columns when the dataset has no suitable sequence structure.
    entity = next((c for c in ("sequence_id", "user_id", "entity_id", "trajectory_id") if c in columns), None)
    event = next((c for c in ("event", "event_type", "user_event", "activity", "user_event_path", "event_path", "events") if c in columns), None)
    time_col = next((c for c in ("time_step", "timestamp", "event_time", "time") if c in columns), "")
    if "trajectory" in low or "trajectory clustering" in low:
        if not entity or not {"latitude", "longitude"}.issubset(columns):
            return {"status": "error", "message": "Trajectory clustering requires an entity/sequence ID plus latitude and longitude columns. Available columns: " + ", ".join(columns)}
        return await call_tool(tm.get("run_trajectory_clustering"), {"entity_col": entity, "latitude_col": "latitude", "longitude_col": "longitude", "n_clusters": 3})
    if not entity:
        return {"status": "error", "message": "Sequential mining requires a sequence/entity ID column. Available columns: " + ", ".join(columns)}
    if not event and "dtw" not in low:
        return {"status": "error", "message": "Sequential pattern analysis requires an event column. Available columns: " + ", ".join(columns)}
    if "dtw" in low:
        value = next((c for c in ("sensor_reading_1", "sensor_reading_2", "value", "reading") if c in columns), None)
        if not time_col or not value:
            return {"status": "error", "message": "DTW requires time and numeric measurement columns. Available columns: " + ", ".join(columns)}
        return await call_tool(tm.get("run_dtw"), {"entity_col": entity, "time_col": time_col, "value_col": value})
    if "prefixspan" in low:
        support = re.search(r"(?:support|min_support)(?:\s+count)?\s*(?:=|of)?\s*(\d+(?:\.\d+)?)", low)
        args = {"entity_col": entity, "event_col": event, "time_col": time_col, "min_support": float(support.group(1)) if support else 0.2}
        return await call_tool(tm.get("run_prefixspan"), args)
    if re.search(r"\bgsp\b", low):
        support = re.search(r"(?:support|min_support)(?:\s+count)?\s*(?:=|of)?\s*(\d+(?:\.\d+)?)", low)
        minimum = float(support.group(1)) if support else 0.2
        return await call_tool(tm.get("run_gsp"), {"entity_col": entity, "event_col": event, "time_col": time_col, "min_support": minimum})
    if "hmm" in low or "hidden markov" in low:
        return await call_tool(tm.get("run_hmm"), {"entity_col": entity, "event_col": event, "time_col": time_col})
    return None


# --------------------------------------------------
# DETERMINISTIC ROUTING
# --------------------------------------------------

async def deterministic_route(text, tools):
    """Execute common requests directly without invoking the LLM."""

    tm = tool_map(tools)
    low = text.lower().strip()

    # Load a dataset
    filename = parse_load(text)

    if filename and "load_dataset" in tm:
        return await call_tool(
            tm["load_dataset"],
            {"filename": filename},
        )

    # Simple dataset operations
    simple = {
        "show columns": "show_columns",
        "show column names": "show_columns",
        "dataset summary": "dataset_summary",
        "summarize the dataset": "dataset_summary",
        "missing values": "missing_values",
        "missing value": "missing_values",
        "duplicates": "duplicate_count",
        "duplicate rows": "duplicate_count",
        "correlation matrix": "correlation_matrix",
    }

    for phrase, tool_name in simple.items():
        if (
            phrase in low
            and tool_name in tm
            and not any(
                word in low
                for word in [
                    "chart", "plot", "graph",
                    "by region", "by category",
                ]
            )
        ):
            return await call_tool(tm[tool_name], {})

    # Display rows
    if (
        ("first" in low or "top" in low) and "row" in low
    ) or "show me the rows" in low:
        if "head" in tm:
            n = min(extract_number(low, 5), 100)
            return await call_tool(tm["head"], {"n": n})

    # Numeric/text filtering: route explicit comparisons directly to the data tool.
    filter_text = re.sub(r"^(?:please\s+)?(?:find|show|list|return)\s+(?:me\s+)?(?:the\s+)?(?:matching\s+)?(?:sales\s+)?records?\s+where\s+", "where ", text, flags=re.IGNORECASE)
    comparison = re.search(r"where\s+([A-Za-z_][\w ]*?)\s*(is\s+greater than|is\s+less than|is\s+at least|is\s+at most|greater than|less than|at least|at most|equal to|equals|is|>=|<=|!=|>|<|==|=)\s*['\"]?([^'\".]+)", filter_text, re.IGNORECASE)
    if comparison and "filter_rows" in tm:
        phrase = comparison.group(1).strip()
        op_text = comparison.group(2).lower().strip()
        value = comparison.group(3).strip()
        op = {"is greater than": ">", "is less than": "<", "is at least": ">=", "is at most": "<=", "greater than": ">", "less than": "<", "at least": ">=", "at most": "<=", "equal to": "=", "equals": "=", "is": "="}.get(op_text, op_text)
        schema = await get_schema(tm)
        names = [x.get("name", "") for x in (schema or {}).get("columns", []) if isinstance(x, dict)]
        column = next((c for c in names if c.lower() == phrase.lower()), None)
        if column:
            return await call_tool(tm["filter_rows"], {"column": column, "operator": op, "value": value, "limit": 5})

    if any(word in low for word in ("prefixspan", "gsp", "hmm", "hidden markov", "dtw", "trajectory", "sequential pattern")):
        return await sequential_route(text, tm)

    # PDF report: pass the actual request so report content follows the user's intent.
    if any(word in low for word in ("report", "pdf")) and "export_pdf_report" in tm:
        title = re.sub(r"\s+", " ", text).strip()
        if len(title) > 100:
            title = title[:97].rstrip() + "..."
        return await call_tool(
            tm["export_pdf_report"],
            {"title": title or "AnalyticsAI Dataset Report", "user_request": text},
        )

    # Spatial analysis
    if any(
        phrase in low
        for phrase in [
            "dbscan",
            "density clustering",
            "gwr",
            "geographically weighted regression",
            "kriging",
            "interpolation",
            "colocation",
            "co-location",
            "spatial outlier",
            "outlier detection",
        ]
    ):
        return await spatial_route(text, tm)

    # Visualizations
    if any(
        phrase in low
        for phrase in ["chart", "plot", "graph", "visualiz"]
    ):
        schema = await get_schema(tm)

        if schema is None:
            return {
                "status": "error",
                "message": "Could not inspect the dataset columns.",
            }

        columns_info = schema.get("columns", [])
        columns = [item["name"] for item in columns_info]
        numeric_columns = [
            item["name"]
            for item in columns_info
            if any(
                dtype in str(item.get("dtype", "")).lower()
                for dtype in ("int", "float", "double")
            )
        ]

        by_match = re.search(
            r"\bby\s+([a-zA-Z][\w ]*)",
            low,
        )

        group_phrase = (
            by_match.group(1).strip(" .?,")
            if by_match
            else (
                "region" if "region" in low
                else "category" if "category" in low
                else ""
            )
        )

        try:
            x_column = (
                resolve_column_locally(columns, group_phrase)
                if group_phrase
                else None
            )
        except Exception:
            x_column = None

        try:
            value_phrase = (
                "sales"
                if "sales" in low or "revenue" in low
                else "quantity"
                if "quantity" in low
                else ""
            )

            y_column = (
                resolve_column_locally(
                    columns,
                    value_phrase,
                    numeric_columns,
                )
                if value_phrase
                else (
                    numeric_columns[0]
                    if numeric_columns
                    else None
                )
            )
        except Exception:
            y_column = (
                numeric_columns[0]
                if numeric_columns
                else None
            )

        if "histogram" in low or "distribution" in low:
            if y_column and "histogram" in tm:
                return await call_tool(
                    tm["histogram"],
                    {"column": y_column},
                )

        if "scatter" in low and len(numeric_columns) >= 2:
            if "scatter_plot" in tm:
                x = numeric_columns[0]
                y = y_column or numeric_columns[1]

                return await call_tool(
                    tm["scatter_plot"],
                    {"x_column": x, "y_column": y},
                )

        if "box" in low and y_column and "box_plot" in tm:
            return await call_tool(
                tm["box_plot"],
                {"column": y_column},
            )

        if (
            "heatmap" in low or "correlation plot" in low
        ) and "correlation_heatmap" in tm:
            return await call_tool(
                tm["correlation_heatmap"],
                {},
            )

        if x_column and y_column:
            aggregation = (
                "mean"
                if "average" in low or "mean" in low
                else "count"
                if "count" in low or "number of" in low
                else "sum"
            )

            if "pie" in low and "pie_chart" in tm:
                return await call_tool(
                    tm["pie_chart"],
                    {
                        "category_column": x_column,
                        "value_column": y_column,
                        "aggregation": aggregation,
                    },
                )

            if "line" in low and "line_chart" in tm:
                return await call_tool(
                    tm["line_chart"],
                    {
                        "x_column": x_column,
                        "y_column": y_column,
                        "aggregation": aggregation,
                    },
                )

            if "bar_chart" in tm:
                return await call_tool(
                    tm["bar_chart"],
                    {
                        "x_column": x_column,
                        "y_column": y_column,
                        "aggregation": aggregation,
                    },
                )

    # Grouped analytical questions
    by_match = re.search(
        r"\bby\s+([a-zA-Z][\w ]*)",
        low,
    )

    if by_match and "group_by_aggregate" in tm:
        group_phrase = by_match.group(1).strip(" .?,")

        aggregation = (
            "mean"
            if "average" in low or "mean" in low
            else "count"
            if "count" in low or "number of" in low
            else "min"
            if "minimum" in low or re.search(r"\bmin\b", low)
            else "max"
            if "maximum" in low or re.search(r"\bmax\b", low)
            else "median"
            if "median" in low
            else "sum"
        )

        schema = await get_schema(tm)

        if schema is not None:
            columns_info = schema.get("columns", [])
            columns = [item["name"] for item in columns_info]
            numeric_columns = [
                item["name"]
                for item in columns_info
                if any(
                    dtype in str(item.get("dtype", "")).lower()
                    for dtype in ("int", "float", "double")
                )
            ]

            try:
                group_column = resolve_column_locally(
                    columns,
                    group_phrase,
                )

                value_phrase = (
                    "sales"
                    if "sales" in low or "revenue" in low
                    else "quantity"
                    if "quantity" in low
                    else low.split("by", 1)[0].strip()
                )

                value_column = resolve_column_locally(
                    columns,
                    value_phrase,
                    numeric_columns,
                )

                return await call_tool(
                    tm["group_by_aggregate"],
                    {
                        "group_column": group_column,
                        "value_column": value_column,
                        "aggregation": aggregation,
                    },
                )

            except Exception:
                pass

    return None


# --------------------------------------------------
# LLM AGENT ROUTING
# --------------------------------------------------

async def llm_route(text, tools):
    group = choose_group(text)

    allowed_names = set(GROUP_TOOLS[group])
    selected_tools = [
        tool for tool in tools
        if tool.name in allowed_names or canonical_tool_name(tool.name) in allowed_names
    ]

    if not selected_tools:
        return {
            "status": "error",
            "message": f"No MCP tools are available for the '{group}' task.",
        }

    llm = make_llm()

    agent = create_agent(
        model=llm,
        tools=selected_tools,
        system_prompt=SYSTEMS[group],
    )

    result = await asyncio.wait_for(
        agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": text,
                    }
                ]
            }
        ),
        timeout=float(os.getenv("AGENT_TIMEOUT", "120")),
    )

    messages = result.get("messages", [])

    # Find the final assistant message rather than assuming the last
    # message always contains a visible answer.
    for message in reversed(messages):
        if getattr(message, "type", "") == "ai":
            content = getattr(message, "content", "")
            # Some chat models return content blocks instead of a plain string.
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict) and isinstance(block.get("text"), str):
                        parts.append(block["text"])
                    elif isinstance(getattr(block, "text", None), str):
                        parts.append(block.text)
                content = "\n".join(parts).strip()
            if isinstance(content, str) and content.strip():
                return content.strip()

    # A tool may have executed successfully while the model omitted its final
    # text. Surface the last tool result rather than losing useful work.
    for message in reversed(messages):
        if getattr(message, "type", "") == "tool":
            content = getattr(message, "content", "")
            if content:
                return {"status": "partial", "message": "The tool executed, but the model did not provide a final explanation.", "tool_result": content}
    return {"status": "error", "message": "The agent returned no final text and no tool result. Check the logged Ollama/MCP exception details."}


# --------------------------------------------------
# RESULT DISPLAY
# --------------------------------------------------

def display_result(result):
    """Display dictionaries, lists, text, and tool results reliably."""

    if result is None:
        result = {
            "status": "error",
            "message": "No result was returned.",
        }

    if isinstance(result, str):
        output = result.strip() or "The assistant returned an empty response."
    else:
        try:
            output = json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        except Exception:
            output = str(result)

    console.print(
        Panel(
            output,
            title="Assistant",
            border_style="cyan",
            expand=True,
        )
    )


# --------------------------------------------------
# MAIN APPLICATION
# --------------------------------------------------

async def main():
    client = MultiServerMCPClient(
        {
            "data_analyst": server("data_server.py"),
            "visualization": server("visualization_server.py"),
            "report": server("report_server.py"),
            "filesystem": server("filesystem_server.py"),
            "sqlite": server("sqlite_server.py"),
            "spatial_sequential": server(
                "spatial_sequential_server.py"
            ),
        }
    )

    try:
        tools = await client.get_tools()
    except Exception as exc:
        console.print(
            Panel(
                f"MCP startup failed:\n\n{type(exc).__name__}: {exc}",
                title="Startup Error",
                border_style="red",
            )
        )
        return

    console.print("\n========== Loaded MCP Tools ==========")

    for tool in tools:
        console.print(f"• {tool.name}")

    console.print("======================================")

    console.print(
        Panel.fit(
            "[bold cyan]AnalyticsAI[/bold cyan]\n"
            "Local AI Data Analyst\n\n"
            "CSV / Excel • SQL • Visualization • Spatial Analysis\n"
            "Sequential Mining • PDF Reports • Files\n\n"
            "Common tasks use direct routing. Other tasks use Qwen3.\n"
            "[yellow]Type 'exit' to quit.[/yellow]",
            title="Welcome",
            border_style="green",
        )
    )

    while True:
        try:
            query = console.input(
                "\n[bold green]You > [/bold green]"
            ).strip()

            if not query:
                continue

            if query.lower() in {"exit", "quit"}:
                break

            with console.status(
                "[bold green]Processing request...[/bold green]",
                spinner="dots",
            ):
                result = await deterministic_route(query, tools)

                if result is None:
                    result = await llm_route(query, tools)

            display_result(result)

        except asyncio.TimeoutError:
            console.print(
                Panel(
                    "The operation exceeded its timeout. "
                    "Check Ollama and MCP server status, then retry.",
                    title="Timeout",
                    border_style="yellow",
                )
            )

        except KeyboardInterrupt:
            console.print("\n[yellow]Request interrupted.[/yellow]")

        except Exception as exc:
            console.print(f"[red]Request failed ({type(exc).__name__}): {exc}[/red]")
            console.print(
                Panel(
                    f"{type(exc).__name__}: {exc}",
                    title="Request Error",
                    border_style="red",
                )
            )


if __name__ == "__main__":
    asyncio.run(main())
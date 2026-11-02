from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd
from fastmcp import FastMCP

from core import get_dataframe, records
from session import session

mcp = FastMCP("AnalyticsAI SQLite Server")
BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "database"
DB_DIR.mkdir(exist_ok=True)
DB_PATH = DB_DIR / "analyticsai.db"


def quote_ident(name: str) -> str:
    return '"' + str(name).replace('"','""') + '"'


def sync_database():
    df = get_dataframe()
    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql("data", conn, if_exists="replace", index=False)
    return len(df)


def connection():
    if not DB_PATH.exists():
        sync_database()
    return sqlite3.connect(DB_PATH)


@mcp.tool
def sync_current_dataset() -> dict:
    try:
        rows=sync_database(); return {"status":"success","table":"data","rows":rows,"database":str(DB_PATH)}
    except Exception as e:return {"status":"error","message":str(e)}


@mcp.tool
def list_tables() -> dict:
    try:
        sync_database()
        with connection() as conn:
            rows=conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
        return {"status":"success","tables":[r[0] for r in rows]}
    except Exception as e:return {"status":"error","message":str(e)}


@mcp.tool
def show_schema(table_name: str = "data") -> dict:
    try:
        sync_database()
        with connection() as conn:
            rows=conn.execute(f"PRAGMA table_info({quote_ident(table_name)})").fetchall()
        return {"status":"success","table":table_name,"columns":[{"name":r[1],"type":r[2],"nullable":not bool(r[3])} for r in rows]}
    except Exception as e:return {"status":"error","message":str(e)}


@mcp.tool
def count_rows(table_name: str = "data") -> dict:
    try:
        sync_database()
        with connection() as conn: n=conn.execute(f"SELECT COUNT(*) FROM {quote_ident(table_name)}").fetchone()[0]
        return {"status":"success","table":table_name,"rows":n}
    except Exception as e:return {"status":"error","message":str(e)}


@mcp.tool
def execute_query(query: str, limit: int = 200) -> dict:
    """Execute a read-only SQL SELECT/CTE query against the current dataset table named data."""
    try:
        q=query.strip().rstrip(";")
        if not re.match(r"^(select|with)\b",q,re.I): raise ValueError("Only SELECT or WITH read-only queries are allowed.")
        sync_database()
        df=pd.read_sql_query(q,connection())
        return {"status":"success","rows":records(df.head(max(1,min(int(limit),5)))),"row_count":len(df),"columns":[str(c) for c in df.columns]}
    except Exception as e:return {"status":"error","message":str(e)}


@mcp.tool
def show_first_rows(table_name: str = "data", limit: int = 5) -> dict:
    try:
        sync_database(); limit=max(1,min(int(limit),100)); df=pd.read_sql_query(f"SELECT * FROM {quote_ident(table_name)} LIMIT {limit}",connection())
        return {"status":"success","rows":records(df)}
    except Exception as e:return {"status":"error","message":str(e)}


if __name__ == "__main__": mcp.run()

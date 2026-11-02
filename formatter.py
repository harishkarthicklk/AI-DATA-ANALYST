from __future__ import annotations
import ast,json
from typing import Any
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
console=Console()

def normalize(x):
    if isinstance(x,str):
        s=x.strip()
        if not s:return ""
        try:return json.loads(s)
        except Exception:
            try:return ast.literal_eval(s)
            except Exception:return x
    if isinstance(x,list):
        out=[]
        for v in x:
            if isinstance(v,dict) and "text" in v:out.append(str(v["text"]))
            else:out.append(str(v))
        return "\n".join(out)
    return x

def display_response(content:Any):
    content=normalize(content)
    if isinstance(content,dict):
        if content.get("status")=="error":console.print(Panel(str(content.get("message",content)),title="Error",border_style="red"));return
        # Render common tabular payloads.
        rows=content.get("rows") or content.get("result") or content.get("counts") or content.get("statistics")
        if isinstance(rows,list) and rows and all(isinstance(r,dict) for r in rows):
            title=content.get("chart_type", "Result").replace("_"," ").title()
            table=Table(title=title,box=box.ROUNDED,show_lines=False)
            keys=list(dict.fromkeys(k for r in rows for k in r.keys()))
            for k in keys:table.add_column(str(k),overflow="fold")
            for r in rows[:100]:table.add_row(*[str(r.get(k,"")) for k in keys])
            console.print(table)
            extras={k:v for k,v in content.items() if k not in {"status","rows","result","counts","statistics"}}
            if extras:console.print(Panel(json.dumps(extras,indent=2,default=str),title="Details",border_style="cyan"))
            return
        console.print(Panel(json.dumps(content,indent=2,default=str),title="Result",border_style="cyan"));return
    console.print(Panel(str(content),title="Assistant",border_style="cyan"))

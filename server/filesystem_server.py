from __future__ import annotations

from pathlib import Path
from fastmcp import FastMCP

mcp=FastMCP("AnalyticsAI Filesystem Server")
BASE=Path(__file__).resolve().parent.parent
ALLOWED={BASE/"datasets",BASE/"charts",BASE/"reports",BASE/"database"}


def safe_path(path:str)->Path:
    p=Path(path)
    if not p.is_absolute(): p=(BASE/p).resolve()
    else:p=p.resolve()
    if any(root==p or root in p.parents for root in ALLOWED): return p
    raise ValueError("Path is outside the project's allowed folders.")

@mcp.tool
def list_files(folder:str=".")->dict:
    try:
        p=safe_path(folder); items=[]
        for x in sorted(p.iterdir()): items.append({"name":x.name,"path":str(x),"type":"directory" if x.is_dir() else "file","size":x.stat().st_size if x.is_file() else None})
        return {"status":"success","files":items}
    except Exception as e:return {"status":"error","message":str(e)}

@mcp.tool
def read_file(path:str)->dict:
    try:
        p=safe_path(path)
        if not p.is_file(): raise ValueError("Not a file.")
        if p.stat().st_size>2_000_000: raise ValueError("File is too large for text reading.")
        return {"status":"success","path":str(p),"content":p.read_text(encoding="utf-8",errors="replace")}
    except Exception as e:return {"status":"error","message":str(e)}

@mcp.tool
def search_files(pattern:str, folder:str=".")->dict:
    try:
        p=safe_path(folder); matches=[str(x) for x in p.rglob(pattern) if x.is_file()]
        return {"status":"success","matches":matches[:500],"count":len(matches)}
    except Exception as e:return {"status":"error","message":str(e)}

@mcp.tool
def write_file(path:str,content:str)->dict:
    try:
        p=safe_path(path); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(content,encoding="utf-8")
        return {"status":"success","path":str(p),"bytes":p.stat().st_size}
    except Exception as e:return {"status":"error","message":str(e)}

@mcp.tool
def file_info(path:str)->dict:
    try:
        p=safe_path(path); s=p.stat(); return {"status":"success","path":str(p),"type":"directory" if p.is_dir() else "file","size":s.st_size}
    except Exception as e:return {"status":"error","message":str(e)}

if __name__=="__main__":mcp.run()

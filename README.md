# AnalyticsAI — rebuilt local AI Data Analyst

This version is rebuilt for **Windows + Python + Ollama + Qwen3:8B**, with no Docker.

## Important design change

The application does **not** send every simple request through Qwen. Common requests are routed directly to the correct MCP tool. This avoids making Qwen inspect dozens of tool schemas for requests such as:

- `Load sales_data.csv`
- `Show me the first 5 rows`
- `Show columns`
- `Give me a dataset summary`
- `What are the total sales by region?`
- `Create a bar chart of total sales by region`
- `Generate a PDF report`

Complex or unusual requests use Qwen, but only with the relevant small MCP tool group. A request timeout prevents an infinite wait.

## Features

- CSV / XLSX / XLS loading
- Dynamic column detection; no sales-dataset column names are required
- Summary, schema, head, descriptive statistics
- Missing values, duplicates, unique values, value counts
- Sum / mean / count / min / max / median grouped analysis
- Filtering and sorting
- Correlation matrix
- Bar / line / pie / histogram / scatter / box / heatmap
- Read-only SQLite over the currently loaded dataset as table `data`
- PDF reports
- Safe project filesystem operations
- DBSCAN, lightweight local weighted regression (not a full MGWR package), spatial co-location, IDW interpolation (not Kriging), spatial outliers
- PrefixSpan projected sequential pattern mining, consecutive-pair GSP-like counting (not full GSP), observed transition probabilities (not a trained HMM), DTW, mean-coordinate trajectory clustering (not full path-shape clustering)
- Tool responses cap representative rows/patterns at five by default; algorithms process the full valid dataset
- Qwen3 fallback for complex natural-language requests

## Install

Use **Python 3.12 (recommended)** for the most predictable Windows wheel compatibility. Python 3.13 may work with current wheels, but if dependency installation fails, recreate the environment with Python 3.12.

```powershell
Expand-Archive .\analyticsai_fixed.zip -DestinationPath .\ -Force
cd .\analyticsai
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --only-binary=:all: -r requirements.txt
Copy-Item .env.example .env -ErrorAction SilentlyContinue
```

Create `.env` from `.env.example` if required.

Check Ollama:

```cmd
ollama list
ollama run qwen3:8b
```

If the model is not installed:

```cmd
ollama pull qwen3:8b
```

Exit the Ollama interactive session with `Ctrl+C` or use another terminal.

## Run

```powershell
.\.venv\Scripts\Activate.ps1
python assistant.py
```

or double-click `start.bat` after creating the virtual environment.

## First validation

Run these in order:

```text
Load spatial_data.csv
Show columns
Run DBSCAN clustering on spatial_data.csv using latitude and longitude.
Create a spatial report using latitude and longitude
Create a correlation report for air_pollution_index and house_value
```

The schema and DBSCAN operations should use the selected dataset directly. Reports now receive the original request, include the requested focus, and choose relevant columns/sections instead of using one fixed generic chart.

You can also run the local smoke test from the project root:

```cmd
python run_tests.py
```

It checks the spatial dataset, schema inspection, DBSCAN, and creation of two distinct request-specific PDF reports. It sets the active dataset to `datasets/spatial_data.csv` while running.

## SQL

The currently loaded dataset is synchronized automatically into SQLite table `data` when a SQL request is executed.

Example:

```text
SELECT Region, SUM(Sales_Amount) AS total_sales
FROM data
GROUP BY Region
ORDER BY total_sales DESC;
```

## Adding another dataset

Put the file into:

```text
datasets/
```

Then:

```text
Load your_file.csv
```

The analysis and visualization code reads the actual schema dynamically.

## Project structure

```text
analyticsai/
├── assistant.py
├── formatter.py
├── create_database.py
├── run_tests.py
├── start.bat
├── .env.example
├── requirements.txt
├── session.json
├── datasets/
├── database/
├── charts/
├── reports/
└── server/
    ├── core.py
    ├── session.py
    ├── data_server.py
    ├── visualization_server.py
    ├── sqlite_server.py
    ├── report_server.py
    ├── filesystem_server.py
    └── spatial_sequential_server.py
```

## Troubleshooting

If a simple request unexpectedly enters `AI analysis...`, the deterministic router did not recognize the wording. The request is then handled by the appropriate smaller Qwen tool group.

If Qwen takes too long, the application stops after `AGENT_TIMEOUT` seconds instead of hanging forever.

If the schema inspection route reports an error even though `Show columns` works, confirm that the MCP tools are returning valid JSON. `assistant.py` now unwraps JSON text content blocks from MCP responses before checking the result status.

Report PDFs and chart images are generated afresh in `reports/` and `charts/`; old generated files are not bundled with this clean source package.

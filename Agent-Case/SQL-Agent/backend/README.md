# SQL-Agent FastAPI backend

This folder contains a small FastAPI wrapper around the existing Streamlit orchestration logic.

How it works
- POST /orchestrate {"query": "..."} — runs the `Orchestrator` from the codebase and returns JSON with either table data, a base64-encoded chart PNG (if orchestration returned a chart), or a raw result and captured logs.

Quick start (local)
1. Create a virtualenv and install requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the server from this folder:

```powershell
python main.py
```

3. Open Swagger UI at: http://127.0.0.1:8000/docs

Notes
- The wrapper imports the project's `Orchestrator` implementation. Ensure dependencies used by the orchestrator (database drivers, transformers, etc.) are installed in the environment.
- Chart results are returned as a base64 PNG string in the `image_base64` field.

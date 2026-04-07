from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict
import sys, os
import io, contextlib, base64
import traceback
import logging

# configure logging: stream to stdout only (no file logging)
logger = logging.getLogger("sql_agent_backend")
if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)


# helper that tees writes to both a buffer and the logger (so prints are logged)
class LoggingTee:
    def __init__(self, logger, level=logging.INFO):
        self.buffer = io.StringIO()
        self.logger = logger
        self.level = level

    def write(self, msg):
        # write to buffer always
        try:
            self.buffer.write(msg)
        except Exception:
            pass
        # log non-empty lines to logger
        text = msg.rstrip('\n')
        if text.strip():
            try:
                self.logger.log(self.level, text)
            except Exception:
                pass

    def flush(self):
        try:
            self.buffer.flush()
        except Exception:
            pass

    def getvalue(self):
        return self.buffer.getvalue()


# Ensure project path is importable (adjust relative to this file)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.append(ROOT)

try:
    # original orchestrator lives under app/orchestrator
    from app.orchestrator.orch_exec import Orchestrator
except Exception as imp_e:
    # fallback if running from repo root
    import_error = None
    try:
        sys.path.append(os.path.abspath(os.path.join(ROOT, '..')))
        from app.orchestrator.orch_exec import Orchestrator
    except Exception as e:
        Orchestrator = None
        import_error = traceback.format_exc()
        # expose import_error for diagnostics
        _IMPORT_ERROR = import_error
        logger.error("Failed to import Orchestrator: %s", import_error)
    else:
        _IMPORT_ERROR = None
else:
    _IMPORT_ERROR = None
    logger.info("Orchestrator imported successfully")

app = FastAPI(title="SQL Orchestrator API", version="0.1")


class QueryRequest(BaseModel):
    query: str


@app.get("/health")
def health():
    return {"status": "ok", "orchestrator_import_error": _IMPORT_ERROR}


@app.post("/orchestrate")
def orchestrate(req: QueryRequest) -> Dict[str, Any]:
    """Run the existing Orchestrator with the provided natural-language query.

    Response shape (examples):
    - table result: {"type":"table","data":[...], "logs": "..."}
    - chart result: {"type":"chart","image_base64":"...","data": [...], "logs": "..."}
    - text/dict: {"type":"raw","data": <value>, "logs": "..."}
    """

    if Orchestrator is None:
        raise HTTPException(status_code=500, detail="Orchestrator import failed. See server logs.")

    user_query = req.query

    stdout_tee = LoggingTee(logger, level=logging.INFO)
    stderr_tee = LoggingTee(logger, level=logging.ERROR)
    try:
        # capture stdout/stderr and forward to logger in real-time via LoggingTee
        with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
            orch = Orchestrator(user_query)
            result = orch.executor()
    except Exception as e:
        logs = stdout_tee.getvalue()
        errs = stderr_tee.getvalue()
        tb = traceback.format_exc()
        logger.exception("Orchestrator execution failed: %s", str(e))
        # Return full traceback in detail for local debugging
        raise HTTPException(status_code=500, detail={"error": str(e), "logs": logs, "stderr": errs, "traceback": tb})

    logs = stdout_tee.getvalue()
    errs = stderr_tee.getvalue()

    # Prepare response depending on result type
    # Avoid returning unserializable types like matplotlib Figure
    try:
        import pandas as pd
    except Exception:
        pd = None

    if pd is not None and hasattr(pd, 'DataFrame') and isinstance(result, pd.DataFrame):
        return {"type": "table", "data": result.to_dict(orient="records"), "logs": logs}

    # Chart path: expect dict with type=="chart" and a matplotlib.figure.Figure under 'figure'
    if isinstance(result, dict) and result.get("type") == "chart":
        fig = result.get("figure")
        data = result.get("data")
        # serialize figure to PNG base64
        try:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode('utf-8')
            return {"type": "chart", "image_base64": img_b64, "data": data.to_dict(orient="records") if pd is not None and hasattr(data, 'to_dict') else data, "logs": logs}
        except Exception as e:
            return {"type": "chart", "error": f"failed to serialize figure: {e}", "data": data, "logs": logs}

    # Generic serializable results (dict/list/str)
    if isinstance(result, (dict, list, str, int, float, bool)):
        return {"type": "raw", "data": result, "logs": logs}

    # Last resort: try converting other objects to string
    return {"type": "unknown", "data": str(result), "logs": logs}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

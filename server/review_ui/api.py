"""Safeline review UI + API.

A standalone FastAPI app (separate process from the voice bot) where an officer
reviews, edits, and approves the AI-generated reports. It reads the same
on-disk report store the bot writes to (``data/incidents``), so nothing extra
needs wiring between the two processes.

Run it::

    uv run python review_ui/api.py
    # or
    uv run uvicorn review_ui.api:app --port 8080

Routes:
    GET  /                                  -> redirect to a hint page
    GET  /review?incident_id=...            -> the review SPA
    GET  /api/incidents                     -> list incident summaries
    GET  /api/reports/{incident_id}         -> full incident (reports + transcript)
    POST /api/reports/{incident_id}/approve -> save officer's edited version + diff
    GET  /api/reports/{incident_id}/diff    -> stored AI-vs-officer diffs (Cekura)
    GET  /api/reports/{incident_id}/evaluate-> run Cekura-style evaluators
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow ``import report_generator`` etc. when run as ``python review_ui/api.py``.
_SERVER_DIR = Path(__file__).resolve().parent.parent
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from dotenv import load_dotenv  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from report_generator import (  # noqa: E402
    approve_report,
    get_diffs,
    get_incident,
    list_incidents,
)

load_dotenv(override=True)

_INDEX_HTML = Path(__file__).resolve().parent / "index.html"

app = FastAPI(title="Safeline Review UI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ApproveBody(BaseModel):
    report_type: str
    fields: dict


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(
        "<h2>Safeline</h2><p>Open a review with "
        "<code>/review?incident_id=INC-...</code></p>"
        '<p><a href="/api/incidents">List incidents</a></p>'
    )


@app.get("/review", response_class=HTMLResponse)
async def review_page():
    if _INDEX_HTML.exists():
        return FileResponse(str(_INDEX_HTML))
    return HTMLResponse("<h1>index.html not found</h1>", status_code=500)


@app.get("/api/incidents")
async def api_list_incidents():
    return {"incidents": list_incidents()}


@app.get("/api/reports/{incident_id}")
async def api_get_reports(incident_id: str):
    record = get_incident(incident_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Unknown incident {incident_id}")
    return record


@app.post("/api/reports/{incident_id}/approve")
async def api_approve(incident_id: str, body: ApproveBody):
    try:
        report = approve_report(incident_id, body.report_type, body.fields)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return {"ok": True, "report": report}


@app.get("/api/reports/{incident_id}/diff")
async def api_diff(incident_id: str):
    record = get_incident(incident_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Unknown incident {incident_id}")
    return {"incident_id": incident_id, "diffs": get_diffs(incident_id)}


@app.get("/api/reports/{incident_id}/evaluate")
async def api_evaluate(incident_id: str):
    record = get_incident(incident_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Unknown incident {incident_id}")
    try:
        from cekura_eval import evaluate_incident

        return JSONResponse(evaluate_incident(record))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {e}")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("REVIEW_UI_PORT", os.getenv("APP_PORT", "8080")))
    uvicorn.run(app, host="0.0.0.0", port=port)

"""Safeline review UI + API.

The voice bot can run separately from this portal, so the bot POSTs generated
reports to ``/api/reports/save`` under the six-digit access code it reads to the
officer. The officer then enters that code in the portal, edits the reports, and
approves them. Approval preserves the immutable AI draft, stores field-level
diffs, and feeds officer corrections into the improvement loop.

Run it::

    uv run python review_ui/api.py
    # or
    uv run uvicorn review_ui.api:app --port 8080
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

_SERVER_DIR = Path(__file__).resolve().parent.parent
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from report_generator import (  # noqa: E402
    approve_report,
    approve_reports as approve_reports_impl,
    get_diffs,
    list_incidents,
    portal_response,
    resolve_incident,
    store_reports,
)

load_dotenv(override=True)

_INDEX_HTML = Path(__file__).resolve().parent / "index.html"

app = FastAPI(title="Safeline Review Portal")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ReportSubmission(BaseModel):
    code: str
    incident_id: str
    officer_badge: str
    officer_name: str | None = None
    reports: dict
    transcript: str = ""
    generated_at: str | None = None


class ApproveBody(BaseModel):
    edited_reports: dict | None = None
    report_type: str | None = None
    fields: dict | None = None


@app.get("/", response_class=HTMLResponse)
async def root():
    return await review_page()


@app.get("/review", response_class=HTMLResponse)
async def review_page():
    if _INDEX_HTML.exists():
        return HTMLResponse(content=_INDEX_HTML.read_text())
    return HTMLResponse("<h1>index.html not found</h1>", status_code=500)


@app.post("/api/reports/save")
async def save_reports(submission: ReportSubmission):
    """Called by the voice bot to save generated reports under a six-digit code."""
    officer = {
        "name": submission.officer_name,
        "badge": submission.officer_badge,
    }
    try:
        record = store_reports(
            submission.incident_id,
            submission.reports,
            transcript=submission.transcript,
            officer=officer,
            officer_badge=submission.officer_badge,
            access_code=submission.code,
        )
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    return {"success": True, "code": submission.code, **portal_response(record)}


@app.get("/api/incidents")
async def api_list_incidents():
    return {"incidents": list_incidents()}


@app.get("/api/reports/{identifier}")
async def api_get_reports(identifier: str):
    record = resolve_incident(identifier)
    if not record:
        raise HTTPException(status_code=404, detail=f"Unknown report code {identifier}")
    return portal_response(record)


@app.post("/api/reports/{identifier}/approve")
async def api_approve(identifier: str, body: ApproveBody):
    try:
        if body.report_type:
            record = resolve_incident(identifier)
            if not record:
                raise KeyError(f"Unknown report code {identifier}")
            edited = body.fields
            if edited is None and body.edited_reports is not None:
                edited = body.edited_reports.get(body.report_type, body.edited_reports)
            report = approve_report(record["incident_id"], body.report_type, edited or {})
            updated = resolve_incident(identifier) or record
            return {"success": True, "report": report, **portal_response(updated)}

        if body.edited_reports is not None:
            record = approve_reports_impl(identifier, body.edited_reports)
            return {"success": True, **portal_response(record)}

        raise ValueError("Provide edited_reports, or report_type and fields")
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.get("/api/reports/{identifier}/diff")
async def api_diff(identifier: str):
    record = resolve_incident(identifier)
    if not record:
        raise HTTPException(status_code=404, detail=f"Unknown report code {identifier}")
    return {"incident_id": record["incident_id"], "diffs": get_diffs(record["incident_id"])}


@app.get("/api/reports/{identifier}/evaluate")
async def api_evaluate(identifier: str):
    record = resolve_incident(identifier)
    if not record:
        raise HTTPException(status_code=404, detail=f"Unknown report code {identifier}")
    try:
        from cekura_eval import evaluate_incident

        return JSONResponse(evaluate_incident(record))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {e}")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("REVIEW_UI_PORT", os.getenv("APP_PORT", "8080")))
    uvicorn.run(app, host="0.0.0.0", port=port)

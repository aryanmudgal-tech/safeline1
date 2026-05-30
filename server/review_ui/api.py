"""Safeline review UI + API (6-digit access-code model).

The voice bot runs on Pipecat Cloud (remote) and this review UI runs locally.
They cannot share memory or files, so the bot POSTs generated reports to this
service's public (ngrok) URL. Reports are keyed by a 6-digit access code that
the bot speaks to the officer; the officer enters that code in the portal to
review, edit, and approve their reports.

Run it::

    uv run python review_ui/api.py
    # or
    uv run uvicorn review_ui.api:app --port 8080

Routes:
    GET  /                          -> the portal SPA (code entry + review)
    POST /api/reports/save          -> bot saves generated reports under a code
    GET  /api/reports/{code}        -> officer portal looks up reports by code
    POST /api/reports/{code}/approve-> officer approves (stores diff + status)
    GET  /api/incidents             -> list all incidents (debugging)
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

_SERVER_DIR = Path(__file__).resolve().parent.parent
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

_TEMPLATES_DIR = _SERVER_DIR / "templates"
_INDEX_HTML = Path(__file__).resolve().parent / "index.html"

app = FastAPI(title="Safeline Review Portal")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store: code -> report data. (Demo storage; swap for a DB in prod.)
REPORTS_STORE: dict[str, dict] = {}


class ReportSubmission(BaseModel):
    code: str
    incident_id: str
    officer_badge: str
    officer_name: str
    reports: dict
    transcript: str
    generated_at: str


class ApprovalSubmission(BaseModel):
    edited_reports: dict
    # Which report was approved (lets the portal track per-report progress).
    # Optional for backwards compatibility; when omitted, all reports are marked.
    report_type: str | None = None


# --------------------------------------------------------------------------- #
# Template labels (loaded locally so the portal can show human-friendly labels)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _templates() -> dict:
    """Return {report_type: [{field, label, priority}, ...]} from templates/."""
    out: dict[str, list] = {}
    if not _TEMPLATES_DIR.exists():
        return out
    for path in _TEMPLATES_DIR.glob("*.json"):
        try:
            with open(path) as f:
                tpl = json.load(f)
            out[tpl["report_type"]] = tpl.get("required_fields", [])
        except Exception:
            continue
    return out


def _field_layout(report_type: str, present_fields: list[str]) -> list[dict]:
    """Ordered [{field, label, priority}] for a report: template order first,
    then any extra fields the model returned that aren't in the template."""
    template = _templates().get(report_type, [])
    layout = []
    seen = set()
    for spec in template:
        f = spec["field"]
        if f in present_fields:
            layout.append(
                {"field": f, "label": spec.get("label", _titleize(f)), "priority": spec.get("priority", 2)}
            )
            seen.add(f)
    for f in present_fields:
        if f not in seen:
            layout.append({"field": f, "label": _titleize(f), "priority": 2})
    return layout


def _titleize(field: str) -> str:
    return field.replace("_", " ").title()


# --------------------------------------------------------------------------- #
# Diff (field-level; clearer for the UI than a generic object diff)
# --------------------------------------------------------------------------- #
def _diff_report(original: dict, edited: dict) -> dict:
    """Return {field: {original, corrected}} for fields that changed."""
    diff = {}
    for f in set(original) | set(edited):
        before = original.get(f)
        after = edited.get(f)
        if (before or None) != (after or None):
            diff[f] = {"original": before, "corrected": after}
    return diff


def _feed_corrections(transcript: str, diffs: dict):
    """Best-effort: push officer edits into the auto-improvement store."""
    if not transcript or not diffs:
        return
    try:
        from improvement.correction_store import correction_store

        for report_type, fields in diffs.items():
            for field, change in fields.items():
                correction_store.add_correction(
                    transcript=transcript,
                    original=change.get("original"),
                    corrected=change.get("corrected"),
                    category=f"{report_type}:{field}",
                    report_type=report_type,
                    field=field,
                    note="Captured from officer portal edit.",
                )
    except Exception:
        # The learning loop must never block an approval.
        pass


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.post("/api/reports/save")
async def save_reports(submission: ReportSubmission):
    """Called by the bot to save generated reports under a 6-digit code."""
    # Deep copy so the immutable AI draft can never be mutated by later edits.
    original = json.loads(json.dumps(submission.reports))
    layouts = {rt: _field_layout(rt, list(fields.keys())) for rt, fields in submission.reports.items()}

    REPORTS_STORE[submission.code] = {
        "incident_id": submission.incident_id,
        "officer_badge": submission.officer_badge,
        "officer_name": submission.officer_name,
        "reports": submission.reports,
        "original_reports": original,
        "field_layouts": layouts,
        "transcript": submission.transcript,
        "generated_at": submission.generated_at,
        "status": "pending_review",
        "approved_report_types": [],
        "diffs": {},
    }
    return {"success": True, "code": submission.code}


@app.get("/api/reports/{code}")
async def get_reports(code: str):
    """Called by the officer portal to look up reports by code."""
    if code not in REPORTS_STORE:
        raise HTTPException(status_code=404, detail="Code not found")
    return REPORTS_STORE[code]


@app.post("/api/reports/{code}/approve")
async def approve_reports(code: str, submission: ApprovalSubmission):
    """Called when an officer approves one (or all) reports."""
    if code not in REPORTS_STORE:
        raise HTTPException(status_code=404, detail="Code not found")

    record = REPORTS_STORE[code]
    original = record["original_reports"]
    edited = submission.edited_reports

    targets = [submission.report_type] if submission.report_type else list(edited.keys())

    diffs = dict(record.get("diffs", {}))
    changed = 0
    for rt in targets:
        d = _diff_report(original.get(rt, {}), edited.get(rt, {}))
        diffs[rt] = d
        changed += len(d)

    approved = set(record.get("approved_report_types", []))
    approved.update(targets)

    record["approved_reports"] = edited
    record["diffs"] = diffs
    record["approved_report_types"] = sorted(approved)
    record["approved_at"] = datetime.now().isoformat()
    total = len(record["reports"])
    record["status"] = "approved" if len(approved) >= total else "partially_approved"

    _feed_corrections(record.get("transcript", ""), {rt: diffs.get(rt, {}) for rt in targets})

    return {
        "success": True,
        "changes": changed,
        "approved_report_types": record["approved_report_types"],
        "status": record["status"],
    }


@app.get("/api/incidents")
async def list_incidents():
    """List all incidents (debugging)."""
    return {
        code: {
            "incident_id": data["incident_id"],
            "officer_name": data["officer_name"],
            "generated_at": data["generated_at"],
            "status": data["status"],
        }
        for code, data in REPORTS_STORE.items()
    }


@app.get("/", response_class=HTMLResponse)
async def root():
    if _INDEX_HTML.exists():
        return HTMLResponse(content=_INDEX_HTML.read_text())
    return HTMLResponse("<h1>index.html not found</h1>", status_code=500)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("REVIEW_UI_PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)

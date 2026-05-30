"""Report generation + permanent report store for Safeline.

Takes the full conversation transcript from a police documentation call and
generates structured reports (incident / arrest / use-of-force / accident) by
extracting fields from natural speech with an LLM.

Two things make this different from Axon Draft One:

  1. PERMANENT AUDIT TRAIL. Every incident is written to ``data/incidents/<id>.json``
     and keeps BOTH the original AI draft (``ai_draft``, immutable) and the
     officer's reviewed/approved version (``officer_version``), plus the diff
     between them. Nothing is ever deleted on export. Disk persistence also
     means the voice bot process and the separate review-UI process share the
     same data.

  2. LEARNS FROM CORRECTIONS. Similar past officer corrections are injected into
     the extraction prompt as few-shot examples (auto-improvement Layer 1).
"""

from __future__ import annotations

import datetime
import json
import os
import secrets
import threading
from pathlib import Path

from loguru import logger
from openai import AsyncOpenAI

_SERVER_DIR = Path(__file__).resolve().parent
_TEMPLATES_DIR = _SERVER_DIR / "templates"
_DATA_DIR = Path(os.getenv("SAFELINE_DATA_DIR", _SERVER_DIR / "data")) / "incidents"

# Metadata keys are stored alongside extracted fields with a leading underscore.
_META_KEYS = ("_report_type", "_officer_badge", "_generated_at", "_status")

_store_lock = threading.Lock()

# In-memory cache; disk (``_DATA_DIR``) is the source of truth so the bot and
# the review API (separate processes) stay consistent.
GENERATED_REPORTS: dict[str, dict] = {}


def _now_iso() -> str:
    return datetime.datetime.now().isoformat()


def _normalize_access_code(code: str | int | None) -> str | None:
    digits = "".join(ch for ch in str(code or "") if ch.isdigit())
    return digits if len(digits) == 6 else None


def _read_incident_file(path: Path) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _write_incident_unlocked(record: dict) -> None:
    incident_id = record["incident_id"]
    GENERATED_REPORTS[incident_id] = record
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(_incident_path(incident_id), "w") as f:
        json.dump(record, f, indent=2)


def _access_code_in_use_unlocked(access_code: str, incident_id: str | None = None) -> bool:
    for rec in GENERATED_REPORTS.values():
        if rec.get("access_code") == access_code and rec.get("incident_id") != incident_id:
            return True

    if _DATA_DIR.exists():
        for path in _DATA_DIR.glob("*.json"):
            rec = _read_incident_file(path)
            if rec and rec.get("access_code") == access_code and rec.get("incident_id") != incident_id:
                return True
    return False


def _generate_access_code_unlocked(incident_id: str | None = None) -> str:
    for _ in range(1000):
        access_code = f"{secrets.randbelow(1_000_000):06d}"
        if not _access_code_in_use_unlocked(access_code, incident_id=incident_id):
            return access_code
    raise RuntimeError("Unable to allocate a unique six-digit access code")


def _record_status(record: dict) -> str:
    if record.get("is_emergency"):
        return "escalated"
    if record.get("generation_error"):
        return "generation_failed"
    reports = record.get("reports") or {}
    if not reports:
        return record.get("status") or "generating"
    if all(report.get("approved") for report in reports.values()):
        return "approved"
    return "pending_review"


def _generated_at(record: dict) -> str | None:
    if record.get("generated_at"):
        return record["generated_at"]
    for report in (record.get("reports") or {}).values():
        if report.get("generated_at"):
            return report["generated_at"]
    return record.get("created_at")


# --------------------------------------------------------------------------- #
# Templates
# --------------------------------------------------------------------------- #
def load_template(report_type: str) -> dict:
    """Load a report template by type (e.g. ``incident_report``)."""
    path = _TEMPLATES_DIR / f"{report_type}.json"
    with open(path) as f:
        return json.load(f)


def _correction_examples(transcript: str, report_type: str) -> str:
    """Few-shot block built from similar past officer corrections (Layer 1)."""
    try:
        from improvement.correction_store import correction_store

        similar = correction_store.get_similar(transcript, k=3, report_type=report_type)
    except Exception as e:  # never let the learning loop break generation
        logger.warning(f"correction_store lookup failed: {e}")
        similar = []

    if not similar:
        return ""

    lines = [
        "\nLEARNED CORRECTIONS — officers previously edited the AI output like this.",
        "Apply the same conventions where relevant (do NOT copy values that aren't in this transcript):",
    ]
    for c in similar:
        field = c.get("field") or "?"
        lines.append(
            f'- field "{field}": prefer phrasing like "{c.get("corrected")}" '
            f'instead of "{c.get("original")}" ({c.get("category", "general")})'
        )
        if c.get("note"):
            lines.append(f"    reason: {c['note']}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
async def generate_reports(
    transcript: str,
    report_types: list[str],
    officer_badge: str,
    client: AsyncOpenAI | None = None,
) -> dict:
    """Generate structured reports from a conversation transcript.

    Returns ``{report_type: {<field>: <value|null>, _report_type, _officer_badge,
    _generated_at, _status}}``. Fields not stated in the transcript are ``null``
    — the model is instructed never to invent information.
    """
    client = client or AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("REPORT_GEN_MODEL", "gpt-4o-mini")

    reports: dict[str, dict] = {}

    for report_type in report_types:
        try:
            template = load_template(report_type)
        except FileNotFoundError:
            logger.warning(f"No template for report_type='{report_type}', skipping")
            continue

        field_specs = template["required_fields"]
        fields = [f["field"] for f in field_specs]
        field_help = "\n".join(f'- {f["field"]}: {f["label"]}' for f in field_specs)

        prompt = f"""You are extracting structured data from a police officer's incident report narrative.

TRANSCRIPT:
{transcript}

Extract the following fields for a {report_type.replace('_', ' ')}.
Return ONLY a JSON object with these exact field names.
If a field was not mentioned, use null.
Do not invent or assume information not explicitly stated by the officer.

Fields to extract:
{field_help}
{_correction_examples(transcript, report_type)}
Return only valid JSON, no explanation."""

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
            )
            extracted = json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"Report generation failed for {report_type}: {e}")
            extracted = {}

        # Normalize: ensure every templated field exists (null if missing).
        clean = {field: extracted.get(field, None) for field in fields}
        clean["_report_type"] = report_type
        clean["_officer_badge"] = officer_badge
        clean["_generated_at"] = datetime.datetime.now().isoformat()
        clean["_status"] = "draft"
        reports[report_type] = clean

    return reports


# --------------------------------------------------------------------------- #
# Persistent store
# --------------------------------------------------------------------------- #
def _incident_path(incident_id: str) -> Path:
    return _DATA_DIR / f"{incident_id}.json"


def _split_fields(report: dict) -> tuple[dict, dict]:
    """Split a generated report into (field_values, metadata)."""
    values = {k: v for k, v in report.items() if not k.startswith("_")}
    meta = {k: v for k, v in report.items() if k.startswith("_")}
    return values, meta


def create_pending_incident(
    incident_id: str,
    *,
    transcript: str | None = None,
    officer: dict | None = None,
    officer_badge: str | None = None,
    report_types: list[str] | None = None,
    access_code: str | int | None = None,
) -> dict:
    """Create or update a portal-visible incident before reports finish.

    The six-digit ``access_code`` is what the voice agent reads on the call.
    While reports are still being generated, GET /api/reports/{code} returns
    this record with ``status="generating"`` and no report payload yet.
    """
    normalized_code = _normalize_access_code(access_code)
    if access_code is not None and not normalized_code:
        raise ValueError("access_code must contain exactly six digits")

    now = _now_iso()
    with _store_lock:
        existing = GENERATED_REPORTS.get(incident_id)
        if not existing:
            path = _incident_path(incident_id)
            existing = _read_incident_file(path) if path.exists() else None

        code = normalized_code or (existing or {}).get("access_code")
        if not code:
            code = _generate_access_code_unlocked(incident_id=incident_id)
        elif _access_code_in_use_unlocked(code, incident_id=incident_id):
            raise ValueError(f"access_code {code} is already in use")

        record = {
            **(existing or {}),
            "incident_id": incident_id,
            "access_code": code,
            "officer_badge": officer_badge or (officer or {}).get("badge") or (existing or {}).get("officer_badge"),
            "officer": officer or (existing or {}).get("officer", {}),
            "transcript": transcript if transcript is not None else (existing or {}).get("transcript", ""),
            "is_emergency": bool((existing or {}).get("is_emergency", False)),
            "created_at": (existing or {}).get("created_at") or now,
            "generated_at": (existing or {}).get("generated_at"),
            "status": "generating",
            "generation_error": None,
            "report_types": report_types or (existing or {}).get("report_types", []),
            "reports": (existing or {}).get("reports", {}),
        }
        _write_incident_unlocked(record)

    logger.info(
        f"Created pending incident {incident_id} with access_code={record['access_code']}"
    )
    return record


def store_reports(
    incident_id: str,
    reports: dict,
    transcript: str | None = None,
    officer: dict | None = None,
    officer_badge: str | None = None,
    is_emergency: bool = False,
    access_code: str | int | None = None,
) -> dict:
    """Persist generated reports for an incident.

    Stores the immutable ``ai_draft`` and a starting ``officer_version`` (equal
    to the draft until the officer edits it), along with the template metadata
    so the review UI can render labels and priorities.
    """
    normalized_code = _normalize_access_code(access_code)
    if access_code is not None and not normalized_code:
        raise ValueError("access_code must contain exactly six digits")

    now = _now_iso()
    with _store_lock:
        existing = GENERATED_REPORTS.get(incident_id)
        if not existing:
            path = _incident_path(incident_id)
            existing = _read_incident_file(path) if path.exists() else None

        code = normalized_code or (existing or {}).get("access_code")
        if not code:
            code = _generate_access_code_unlocked(incident_id=incident_id)
        elif _access_code_in_use_unlocked(code, incident_id=incident_id):
            raise ValueError(f"access_code {code} is already in use")

        record = {
            "incident_id": incident_id,
            "access_code": code,
            "officer_badge": officer_badge or (officer or {}).get("badge") or (existing or {}).get("officer_badge"),
            "officer": officer or (existing or {}).get("officer", {}),
            "transcript": transcript if transcript is not None else (existing or {}).get("transcript", ""),
            "is_emergency": is_emergency,
            "created_at": (existing or {}).get("created_at") or now,
            "generated_at": now,
            "status": "pending_review",
            "generation_error": None,
            "report_types": list(reports.keys()),
            "reports": {},
        }

        for report_type, report in reports.items():
            values, meta = _split_fields(report)
            try:
                template = load_template(report_type)["required_fields"]
            except FileNotFoundError:
                template = [{"field": k, "label": k, "priority": 2} for k in values]
            record["reports"][report_type] = {
                "_report_type": report_type,
                "template": template,
                "ai_draft": dict(values),
                "officer_version": dict(values),
                "approved": False,
                "approved_at": None,
                "diff": {},
                "generated_at": meta.get("_generated_at") or now,
                "status": "draft",
            }

        _write_incident_unlocked(record)

    logger.info(
        f"Stored incident {incident_id}: reports={list(record['reports'].keys())} "
        f"-> {_incident_path(incident_id)}"
    )
    return record


def get_incident(incident_id: str) -> dict | None:
    """Return the full incident record (reports + transcript + metadata)."""
    with _store_lock:
        if incident_id in GENERATED_REPORTS:
            return GENERATED_REPORTS[incident_id]
        path = _incident_path(incident_id)
        if path.exists():
            with open(path) as f:
                record = json.load(f)
            GENERATED_REPORTS[incident_id] = record
            return record
    return None


def get_incident_by_access_code(access_code: str | int) -> dict | None:
    """Return an incident by its six-digit portal access code."""
    normalized = _normalize_access_code(access_code)
    if not normalized:
        return None

    with _store_lock:
        for record in GENERATED_REPORTS.values():
            if record.get("access_code") == normalized:
                return record

        if _DATA_DIR.exists():
            for path in _DATA_DIR.glob("*.json"):
                record = _read_incident_file(path)
                if not record:
                    continue
                GENERATED_REPORTS[record["incident_id"]] = record
                if record.get("access_code") == normalized:
                    return record
    return None


def resolve_incident(identifier: str | int) -> dict | None:
    """Resolve either a six-digit access code or a legacy incident id."""
    return get_incident_by_access_code(identifier) or get_incident(str(identifier))


def get_reports(incident_id: str) -> dict:
    """Return just the reports map for an incident (spec-compatible helper)."""
    record = get_incident(incident_id)
    return record.get("reports", {}) if record else {}


def list_incidents() -> list[dict]:
    """List incident summaries (newest first), reading from disk."""
    summaries = []
    if _DATA_DIR.exists():
        for path in _DATA_DIR.glob("*.json"):
            try:
                with open(path) as f:
                    rec = json.load(f)
                summaries.append(
                    {
                        "incident_id": rec.get("incident_id"),
                        "access_code": rec.get("access_code"),
                        "officer_badge": rec.get("officer_badge"),
                        "created_at": rec.get("created_at"),
                        "generated_at": _generated_at(rec),
                        "status": _record_status(rec),
                        "report_types": rec.get("report_types", []),
                        "approved": all(
                            r.get("approved") for r in rec.get("reports", {}).values()
                        ),
                    }
                )
            except Exception:
                continue
    summaries.sort(key=lambda s: s.get("created_at") or "", reverse=True)
    return summaries


def _compute_diff(ai_draft: dict, officer_version: dict) -> dict:
    """Field-level diff: {field: {original, corrected}} for changed fields."""
    diff = {}
    keys = set(ai_draft) | set(officer_version)
    for k in keys:
        before = ai_draft.get(k)
        after = officer_version.get(k)
        if (before or None) != (after or None):
            diff[k] = {"original": before, "corrected": after}
    return diff


def approve_report(incident_id: str, report_type: str, edited_fields: dict) -> dict:
    """Save the officer's reviewed version of one report, compute + store the
    diff vs. the AI draft, mark it approved, and feed the diff back into the
    correction store (auto-improvement Layer 1).
    """
    record = get_incident(incident_id)
    if not record:
        raise KeyError(f"Unknown incident {incident_id}")
    if report_type not in record["reports"]:
        raise KeyError(f"Incident {incident_id} has no report '{report_type}'")

    report = record["reports"][report_type]
    ai_draft = report["ai_draft"]

    officer_version = dict(report["officer_version"])
    officer_version.update(edited_fields or {})

    diff = _compute_diff(ai_draft, officer_version)

    report["officer_version"] = officer_version
    report["diff"] = diff
    report["approved"] = True
    report["approved_at"] = datetime.datetime.now().isoformat()
    report["status"] = "approved"
    record["status"] = _record_status(record)
    if record["status"] == "approved":
        record["approved_at"] = _now_iso()

    with _store_lock:
        _write_incident_unlocked(record)

    _feed_corrections(record.get("transcript", ""), report_type, diff)

    logger.info(f"Approved {report_type} for {incident_id} with {len(diff)} edits")
    return report


def approve_reports(identifier: str | int, edited_reports: dict) -> dict:
    """Approve one or more reports from the portal's bulk edit payload."""
    record = resolve_incident(identifier)
    if not record:
        raise KeyError(f"Unknown report code or incident id {identifier}")
    if _record_status(record) in {"generating", "generation_failed", "escalated"}:
        raise ValueError(f"Incident {record['incident_id']} is not ready for approval")
    if not edited_reports:
        raise ValueError("edited_reports must include at least one report")

    for report_type, fields in edited_reports.items():
        if report_type not in record.get("reports", {}):
            raise KeyError(f"Incident {record['incident_id']} has no report '{report_type}'")
        approve_report(record["incident_id"], report_type, fields or {})

    return get_incident(record["incident_id"]) or record


def mark_generation_failed(incident_id: str, error: str) -> dict | None:
    """Move a pending incident into a visible failure state."""
    with _store_lock:
        record = GENERATED_REPORTS.get(incident_id)
        if not record:
            path = _incident_path(incident_id)
            record = _read_incident_file(path) if path.exists() else None
        if not record:
            return None
        record["status"] = "generation_failed"
        record["generation_error"] = error
        _write_incident_unlocked(record)
        return record


def portal_response(record: dict) -> dict:
    """External API shape for the six-digit-code web portal."""
    officer = record.get("officer") or {}
    reports = record.get("reports") or {}
    flat_reports = {
        report_type: dict(report.get("officer_version") or report.get("ai_draft") or {})
        for report_type, report in reports.items()
    }
    ai_drafts = {
        report_type: dict(report.get("ai_draft") or {}) for report_type, report in reports.items()
    }
    report_templates = {
        report_type: report.get("template", []) for report_type, report in reports.items()
    }
    approvals = {
        report_type: {
            "approved": bool(report.get("approved")),
            "approved_at": report.get("approved_at"),
            "diff": report.get("diff", {}),
        }
        for report_type, report in reports.items()
    }

    return {
        "incident_id": record.get("incident_id"),
        "access_code": record.get("access_code"),
        "officer_name": officer.get("name"),
        "officer_badge": record.get("officer_badge") or officer.get("badge"),
        "created_at": record.get("created_at"),
        "generated_at": _generated_at(record),
        "status": _record_status(record),
        "generation_error": record.get("generation_error"),
        "reports": flat_reports,
        "report_templates": report_templates,
        "ai_draft": ai_drafts,
        "approvals": approvals,
        "report_types": record.get("report_types", list(flat_reports.keys())),
        "transcript": record.get("transcript", ""),
        "is_emergency": bool(record.get("is_emergency")),
        "escalation_reason": record.get("escalation_reason"),
    }


def _feed_corrections(transcript: str, report_type: str, diff: dict):
    """Push each officer edit into the correction store as a learning example."""
    if not diff:
        return
    try:
        from improvement.correction_store import correction_store

        for field, change in diff.items():
            correction_store.add_correction(
                transcript=transcript,
                original=change.get("original"),
                corrected=change.get("corrected"),
                category=f"{report_type}:{field}",
                report_type=report_type,
                field=field,
                note="Captured from officer review edit.",
            )
    except Exception as e:
        logger.warning(f"Could not feed corrections to store: {e}")


def get_diffs(incident_id: str) -> dict:
    """Return stored AI-draft-vs-officer diffs for every report (for Cekura)."""
    record = get_incident(incident_id)
    if not record:
        return {}
    return {
        rt: {
            "approved": r.get("approved"),
            "diff": r.get("diff", {}),
        }
        for rt, r in record.get("reports", {}).items()
    }

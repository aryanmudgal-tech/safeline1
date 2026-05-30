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


def store_reports(
    incident_id: str,
    reports: dict,
    transcript: str | None = None,
    officer: dict | None = None,
    officer_badge: str | None = None,
    is_emergency: bool = False,
) -> dict:
    """Persist generated reports for an incident.

    Stores the immutable ``ai_draft`` and a starting ``officer_version`` (equal
    to the draft until the officer edits it), along with the template metadata
    so the review UI can render labels and priorities.
    """
    record = {
        "incident_id": incident_id,
        "officer_badge": officer_badge or (officer or {}).get("badge"),
        "officer": officer or {},
        "transcript": transcript or "",
        "is_emergency": is_emergency,
        "created_at": datetime.datetime.now().isoformat(),
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
            "generated_at": meta.get("_generated_at"),
            "status": "draft",
        }

    with _store_lock:
        GENERATED_REPORTS[incident_id] = record
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(_incident_path(incident_id), "w") as f:
            json.dump(record, f, indent=2)

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
                        "officer_badge": rec.get("officer_badge"),
                        "created_at": rec.get("created_at"),
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

    with _store_lock:
        GENERATED_REPORTS[incident_id] = record
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(_incident_path(incident_id), "w") as f:
            json.dump(record, f, indent=2)

    _feed_corrections(record.get("transcript", ""), report_type, diff)

    logger.info(f"Approved {report_type} for {incident_id} with {len(diff)} edits")
    return report


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

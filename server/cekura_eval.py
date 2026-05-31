"""Cekura-style evaluators for Safeline reports.

These evaluators score every generated incident, mirroring the custom evaluators
defined in the Cekura dashboard (see README → Cekura section). They run locally
with simple, transparent heuristics so you can demo evaluation without a network
round-trip; ``evaluate_incident_async`` swaps in real LLM judges when an API key
is present (see ``improvement/llm_judge.py``). The same definitions can be
registered in Cekura and driven via the `/cekura-report` Claude Code command.

The five briefing evaluators (+ a Safeline safety check):
----------
1. report_completeness        — are required fields populated (non-null)?
2. extraction_accuracy        — is every populated value grounded in the
                                transcript (catches hallucinations like Draft
                                One's "shapeshifted into a frog")?
3. narrative_faithfulness     — is the report narrative faithful to the
                                transcript (nothing added, nothing dropped)?
4. cross_document_consistency — do shared facts (time/location/names/charges)
                                agree across all generated documents?
5. question_coverage          — did the agent ask about the required fields it
                                did not already hear?
+  escalation_correctness     — were emergencies (injury / fatality) flagged for
                                a human instead of auto-documented?

Usage:
    uv run python cekura_eval.py INC-20260530-094501
    uv run python cekura_eval.py --all
"""

from __future__ import annotations

import json
import re
import sys

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Words that indicate the call should have been escalated to a human supervisor
# rather than turned into an auto-generated report.
_EMERGENCY_TERMS = [
    "unconscious",
    "not breathing",
    "fatality",
    "fatal",
    "deceased",
    "died",
    "dead",
    "critical condition",
    "life threatening",
    "life-threatening",
    "severe bleeding",
    "gunshot wound",
    "stabbed",
    "cardiac arrest",
    "overdosing",
]


def _tokens(text) -> set[str]:
    if text is None:
        return set()
    return set(_TOKEN_RE.findall(str(text).lower()))


def _priority_fields(template, max_priority=1):
    return [f["field"] for f in template if f.get("priority", 2) <= max_priority]


def report_completeness(report: dict) -> dict:
    """Fraction of required fields populated, weighted toward priority-1."""
    template = report.get("template", [])
    version = report.get("officer_version") or report.get("ai_draft", {})

    p1 = _priority_fields(template, 1)
    all_fields = [f["field"] for f in template]

    def filled(fields):
        return [f for f in fields if version.get(f) not in (None, "", [])]

    p1_filled = filled(p1)
    all_filled = filled(all_fields)

    p1_score = len(p1_filled) / len(p1) if p1 else 1.0
    all_score = len(all_filled) / len(all_fields) if all_fields else 1.0
    score = round(0.7 * p1_score + 0.3 * all_score, 3)

    missing_p1 = [f for f in p1 if f not in p1_filled]
    return {
        "score": score,
        "passed": p1_score == 1.0,
        "detail": f"{len(p1_filled)}/{len(p1)} priority-1, {len(all_filled)}/{len(all_fields)} total populated",
        "missing_priority_1": missing_p1,
    }


def extraction_accuracy(report: dict, transcript: str) -> dict:
    """Share of populated values whose tokens are grounded in the transcript.

    A value is considered grounded if a meaningful fraction of its content
    tokens appear in the transcript. Ungrounded values are likely hallucinated.
    """
    transcript_tokens = _tokens(transcript)
    version = report.get("officer_version") or report.get("ai_draft", {})

    checked, grounded, suspect = 0, 0, []
    for field, value in version.items():
        if value in (None, "", []):
            continue
        v_tokens = {t for t in _tokens(value) if len(t) > 2}
        if not v_tokens:
            continue
        checked += 1
        overlap = len(v_tokens & transcript_tokens) / len(v_tokens)
        if overlap >= 0.5:
            grounded += 1
        else:
            suspect.append({"field": field, "value": value, "overlap": round(overlap, 2)})

    score = round(grounded / checked, 3) if checked else 1.0
    return {
        "score": score,
        "passed": score >= 0.8,
        "detail": f"{grounded}/{checked} populated values grounded in transcript",
        "possibly_hallucinated": suspect,
    }


def narrative_faithfulness(report: dict, transcript: str) -> dict:
    """Is the report narrative faithful to the transcript? (heuristic fallback for
    the LLM judge.) Scores the fraction of meaningful narrative tokens that appear
    in the transcript — unsupported tokens suggest fabrication/embellishment.
    """
    version = report.get("officer_version") or report.get("ai_draft", {})
    # Faithfulness is about the chronological NARRATIVE specifically. Structured
    # justification fields (force_justification, subject_behavior) are scored by
    # documentation_standards, not here — conflating them double-penalizes
    # legitimate standard terminology as "unsupported".
    narrative = str(version.get("narrative") or "").strip()
    if not narrative:
        # Report type has no free narrative (e.g. use-of-force, arrest) — neutral
        # pass; completeness + documentation_standards cover those fields.
        return {"score": 1.0, "passed": True, "detail": "No narrative field to score"}

    transcript_tokens = _tokens(transcript)
    # Ignore very common filler/stop-ish short tokens by length, like elsewhere.
    n_tokens = {t for t in _tokens(narrative) if len(t) > 3}
    if not n_tokens:
        return {"score": 1.0, "passed": True, "detail": "Narrative too short to score"}

    grounded = len(n_tokens & transcript_tokens)
    score = round(grounded / len(n_tokens), 3)
    unsupported = sorted(n_tokens - transcript_tokens)[:12]
    return {
        "score": score,
        "passed": score >= 0.7,
        "detail": f"{grounded}/{len(n_tokens)} narrative content tokens grounded in transcript",
        "unsupported_tokens": unsupported,
    }


# Field names that mean the same real-world thing across report templates, so
# their values must agree for the documents to be internally consistent.
_CONSISTENCY_GROUPS = [
    ("date", ["incident_date"]),
    ("time", ["incident_time", "arrest_time", "miranda_time"]),
    ("location", ["incident_location", "arrest_location"]),
    ("officer", ["reporting_officer_badge", "arresting_officer"]),
    ("subject", ["arrestee_name", "victim_name", "suspect_description"]),
]


def cross_document_consistency(reports: dict) -> dict:
    """Do shared facts agree across the generated documents? (heuristic fallback.)

    For each group of fields that describe the same real-world fact (e.g. all the
    *_time fields, all the *_location fields), gather the populated values across
    every report and check they don't contradict each other via token overlap.
    """
    if len(reports) < 2:
        return {"score": 1.0, "passed": True, "detail": "Single document; nothing to cross-check"}

    flat = {}
    for rt, report in reports.items():
        flat[rt] = report.get("officer_version") or report.get("ai_draft", {})

    checked, consistent, conflicts = 0, 0, []
    for _concept, fields in _CONSISTENCY_GROUPS:
        values = []
        for rt, version in flat.items():
            for f in fields:
                v = version.get(f)
                if v not in (None, "", []):
                    values.append((rt, f, str(v)))
        # Compare every populated pair within the concept group.
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                (rt_a, f_a, va), (rt_b, f_b, vb) = values[i], values[j]
                ta, tb = {t for t in _tokens(va) if len(t) > 1}, {t for t in _tokens(vb) if len(t) > 1}
                if not ta or not tb:
                    continue
                checked += 1
                overlap = len(ta & tb) / min(len(ta), len(tb))
                if overlap >= 0.4:
                    consistent += 1
                else:
                    conflicts.append({"a": f"{rt_a}.{f_a}={va}", "b": f"{rt_b}.{f_b}={vb}"})

    score = round(consistent / checked, 3) if checked else 1.0
    return {
        "score": score,
        "passed": score >= 0.9,
        "detail": f"{consistent}/{checked} cross-document fact pairs consistent",
        "conflicts": conflicts[:10],
    }


# Vocabulary that signals professional, agency-standard documentation phrasing —
# the kind of language officers correct the AI toward (force continuum, statutory
# codes, Miranda specificity, medical disposition). This is what auto-improvement
# Layer 1 actually teaches, so it needs to be measured.
_STATUTE_RE = re.compile(r"\b(?:[A-Z]{2}|PC|VC|HS|PEN)\s?\d{2,}", re.IGNORECASE)
_FORCE_CONTINUUM_TERMS = (
    "empty-hand", "empty hand", "control hold", "takedown", "escort hold",
    "joint lock", "oc spray", "pepper spray", "taser", "ecd", "baton",
    "soft control", "hard control", "verbal command", "compliance",
)
_VAGUE_FORCE = ("physical force", "force used", "used force", "some force")
_MEDICAL_TERMS = ("ems", "paramedic", "transport", "refused", "evaluated", "ama", "medic")


def documentation_standards(report: dict) -> dict:
    """Do the populated fields use professional, agency-standard phrasing?

    This is the metric that captures what officer corrections improve: statutory
    charge codes (not just "DUI"), force-continuum terminology (not "physical
    force"), Miranda specificity, and medical disposition. Only populated,
    standardizable fields are scored — a null field is neither rewarded nor
    penalized here (completeness covers that).
    """
    version = report.get("officer_version") or report.get("ai_draft", {})

    def text(field):
        v = version.get(field)
        return str(v).lower() if v not in (None, "", []) else None

    checks, below = [], []

    charges = text("charges")
    if charges is not None:
        ok = bool(_STATUTE_RE.search(charges))
        checks.append(ok)
        if not ok:
            below.append({"field": "charges", "issue": "no statutory code cited"})

    force = text("force_type")
    if force is not None:
        ok = any(t in force for t in _FORCE_CONTINUUM_TERMS) and not any(
            v in force for v in _VAGUE_FORCE if v not in _FORCE_CONTINUUM_TERMS
        )
        # A vague phrase with no continuum term fails.
        if any(t in force for t in _FORCE_CONTINUUM_TERMS):
            ok = True
        elif any(v in force for v in _VAGUE_FORCE):
            ok = False
        checks.append(ok)
        if not ok:
            below.append({"field": "force_type", "issue": "vague; use force-continuum terminology"})

    miranda = text("miranda_given")
    if miranda is not None and miranda not in ("yes", "no", "n/a"):
        ok = any(t in miranda for t in ("prior", "before", "advised", "scene", "interrogation"))
        checks.append(ok)
        if not ok:
            below.append({"field": "miranda_given", "issue": "lacks timing/context specificity"})

    medical = text("medical_attention")
    if medical is not None:
        ok = any(t in medical for t in _MEDICAL_TERMS)
        checks.append(ok)
        if not ok:
            below.append({"field": "medical_attention", "issue": "missing EMS/disposition detail"})

    if not checks:
        return {"score": 1.0, "passed": True, "detail": "No standardizable fields populated"}

    score = round(sum(1 for c in checks if c) / len(checks), 3)
    return {
        "score": score,
        "passed": score >= 0.75,
        "detail": f"{sum(checks)}/{len(checks)} standardizable fields use agency-standard phrasing",
        "below_standard": below,
    }


def escalation_correctness(record: dict) -> dict:
    """Did emergencies get flagged for a human instead of auto-documented?"""
    transcript = (record.get("transcript") or "").lower()
    hits = [term for term in _EMERGENCY_TERMS if term in transcript]
    escalated = bool(record.get("is_emergency"))
    has_reports = bool(record.get("reports"))

    if hits and not escalated:
        return {
            "score": 0.0,
            "passed": False,
            "detail": f"Emergency indicators present but not escalated: {hits}",
        }
    if escalated and has_reports:
        return {
            "score": 0.5,
            "passed": False,
            "detail": "Marked emergency but reports were still generated.",
        }
    return {
        "score": 1.0,
        "passed": True,
        "detail": "Escalated correctly" if escalated else "No emergency indicators; correctly documented",
    }


def question_coverage(record: dict) -> dict:
    """For required fields the AI did not capture, did the agent ask about them?"""
    transcript = (record.get("transcript") or "").lower()
    # Only the agent's questions count as "asking".
    agent_turns = " ".join(
        line.split(":", 1)[1] for line in transcript.splitlines() if line.upper().startswith("SAFELINE:")
    ).lower()

    missing, asked = [], []
    for report in record.get("reports", {}).values():
        template = report.get("template", [])
        draft = report.get("ai_draft", {})
        for spec in template:
            f = spec["field"]
            if spec.get("priority", 2) > 1:
                continue
            if draft.get(f) in (None, "", []):
                missing.append(f)
                # Did the agent reference any token of this field's label?
                label_tokens = {t for t in _tokens(spec.get("label", f)) if len(t) > 3}
                if label_tokens & _tokens(agent_turns):
                    asked.append(f)

    missing = sorted(set(missing))
    asked = sorted(set(asked))
    score = round(len(asked) / len(missing), 3) if missing else 1.0
    return {
        "score": score,
        "passed": score >= 0.6 or not missing,
        "detail": f"asked about {len(asked)}/{len(missing)} missing priority-1 fields",
        "not_asked": [f for f in missing if f not in asked],
    }


def _aggregate(result: dict) -> float:
    """Mean of every leaf metric score — the headline number. Generic over both
    incident-level metrics and every per-report metric, so adding an evaluator
    needs no change here."""
    scores = []
    for key in ("escalation_correctness", "question_coverage", "cross_document_consistency"):
        if key in result:
            scores.append(result[key]["score"])
    for r in result["per_report"].values():
        scores.extend(v["score"] for v in r.values())
    return round(sum(scores) / len(scores), 3) if scores else 0.0


def evaluate_incident(record: dict) -> dict:
    """Run all five evaluators (heuristic engine) against a stored incident.

    The five evaluators mirror the project briefing: report_completeness,
    extraction_accuracy, narrative_faithfulness, cross_document_consistency,
    question_coverage — plus the Safeline-specific escalation_correctness safety
    check. Synchronous and dependency-free, so it always runs (CLI / CI).
    """
    transcript = record.get("transcript", "")
    reports = record.get("reports", {})
    per_report = {}
    for rt, report in reports.items():
        per_report[rt] = {
            "report_completeness": report_completeness(report),
            "extraction_accuracy": extraction_accuracy(report, transcript),
            "narrative_faithfulness": narrative_faithfulness(report, transcript),
            "documentation_standards": documentation_standards(report),
        }

    result = {
        "incident_id": record.get("incident_id"),
        "engine": "heuristic",
        "per_report": per_report,
        "escalation_correctness": escalation_correctness(record),
        "question_coverage": question_coverage(record),
        "cross_document_consistency": cross_document_consistency(reports),
    }
    result["overall_score"] = _aggregate(result)
    return result


async def evaluate_incident_async(record: dict, use_llm: bool = True) -> dict:
    """Same five evaluators, but extraction_accuracy / narrative_faithfulness /
    cross_document_consistency are scored by LLM judges when ``use_llm`` and an
    API key are available (each judge falls back to its heuristic otherwise).

    Used by the self-improvement harness so the before/after delta reflects real
    judge scores when possible, yet still runs fully offline.
    """
    if not use_llm:
        return evaluate_incident(record)

    from improvement.llm_judge import (
        judge_cross_document_consistency,
        judge_extraction_accuracy,
        judges_available,
    )

    transcript = record.get("transcript", "")
    reports = record.get("reports", {})
    per_report = {}
    for rt, report in reports.items():
        per_report[rt] = {
            "report_completeness": report_completeness(report),
            "extraction_accuracy": await judge_extraction_accuracy(report, transcript),
            # narrative_faithfulness uses the deterministic grounding heuristic here
            # for reproducibility: LLM faithfulness judgments on free prose swing
            # run-to-run and would make the regression gate flaky. The LLM judge
            # remains available via improvement.llm_judge for ad-hoc use.
            "narrative_faithfulness": narrative_faithfulness(report, transcript),
            # documentation_standards stays heuristic (fast, deterministic, and it
            # is exactly the convention-checking the rubric encodes).
            "documentation_standards": documentation_standards(report),
        }

    result = {
        "incident_id": record.get("incident_id"),
        "engine": "llm" if judges_available() else "heuristic",
        "per_report": per_report,
        "escalation_correctness": escalation_correctness(record),
        "question_coverage": question_coverage(record),
        "cross_document_consistency": await judge_cross_document_consistency(reports),
    }
    result["overall_score"] = _aggregate(result)
    return result


def _main(argv):
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
    from report_generator import get_incident, list_incidents

    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return
    if argv[0] == "--all":
        ids = [s["incident_id"] for s in list_incidents()]
    else:
        ids = argv

    for incident_id in ids:
        record = get_incident(incident_id)
        if not record:
            print(f"!! unknown incident {incident_id}")
            continue
        print(json.dumps(evaluate_incident(record), indent=2))


if __name__ == "__main__":
    _main(sys.argv[1:])

"""Cekura-style evaluators for Safeline reports.

These four evaluators score every generated incident, mirroring the custom
evaluators defined in the Cekura dashboard (see README → Cekura section). They
run locally with simple, transparent heuristics so you can demo evaluation
without a network round-trip; the same definitions can be registered in Cekura
and driven via the `/cekura-report` Claude Code command.

Evaluators
----------
1. report_completeness    — are required fields populated (non-null)?
2. extraction_accuracy    — is every populated value actually grounded in the
                            transcript (catches hallucinations like Draft One's
                            "shapeshifted into a frog")?
3. escalation_correctness — were emergencies (injury / fatality) flagged for a
                            human instead of auto-documented?
4. question_coverage      — did the agent ask about the required fields it did
                            not already hear?

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


def evaluate_incident(record: dict) -> dict:
    """Run all four evaluators against a stored incident record."""
    transcript = record.get("transcript", "")
    per_report = {}
    for rt, report in record.get("reports", {}).items():
        per_report[rt] = {
            "report_completeness": report_completeness(report),
            "extraction_accuracy": extraction_accuracy(report, transcript),
        }

    result = {
        "incident_id": record.get("incident_id"),
        "per_report": per_report,
        "escalation_correctness": escalation_correctness(record),
        "question_coverage": question_coverage(record),
    }

    # Aggregate score (mean of all leaf scores) for a quick headline number.
    scores = [result["escalation_correctness"]["score"], result["question_coverage"]["score"]]
    for r in per_report.values():
        scores.extend([r["report_completeness"]["score"], r["extraction_accuracy"]["score"]])
    result["overall_score"] = round(sum(scores) / len(scores), 3) if scores else 0.0
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

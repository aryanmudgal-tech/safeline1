"""Cekura integration layer for the auto-improvement loop (Layer 2, hosted).

This module is the *code* side of the Cekura wiring. The actual creation of
metrics / scenarios / runs / dashboards is done through the Cekura MCP server
(driven from Claude Code), but everything those calls need is defined here as data
and helpers so the integration is reproducible and version-controlled:

  * ``METRIC_DEFINITIONS`` — the five evaluators as Cekura metric specs (the same
    contracts ``cekura_eval`` / ``llm_judge`` implement locally), ready to feed
    ``metrics_bulk_create``.
  * ``corrections_to_scenarios`` — turn every officer/teacher correction into a
    Cekura regression scenario. This operationalizes the briefing's "every officer
    correction becomes a Cekura test case", so the suite grows as the agent is
    used and old mistakes can never silently regress.
  * ``red_team_scenarios`` — adversarial "hallucination" cases (the Axon Draft One
    "shapeshifted into a frog" failure mode) asserting the agent emits null instead
    of inventing.
  * ``push_experiment`` — serialize a self-improvement run into Cekura-call-log
    payloads (baseline batch + improved batch) so the lift can be scored on the
    dashboard, and write them next to the experiment artifacts.
  * ``fetch_run_scores`` — best-effort fetch of a Cekura run's per-scenario scores
    for ``regression_gate.check_via_cekura`` (no-ops gracefully without creds).

Novel uses folded in (see README): corrections→scenarios, Cekura-as-regression-gate,
judge tuning from officer corrections (feed corrections as metric feedback), and a
baseline-vs-improved dashboard.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from loguru import logger

# --------------------------------------------------------------------------- #
# The five metric specifications (mirror cekura_eval / llm_judge)
# --------------------------------------------------------------------------- #
METRIC_DEFINITIONS = [
    {
        "name": "report_completeness",
        "type": "deterministic",
        "description": "Fraction of required report fields populated (priority-1 weighted).",
        "passing_threshold": 1.0,
        "evaluation": "All priority-1 template fields must be non-null.",
    },
    {
        "name": "extraction_accuracy",
        "type": "llm_judge",
        "description": "Are extracted values grounded in the transcript (no hallucination)?",
        "passing_threshold": 0.8,
        "judge_prompt": (
            "Score how well each extracted field value is grounded in the officer's "
            "transcript. Any invented, assumed, or contradicted value sharply reduces "
            "the score. Return score 0..1."
        ),
    },
    {
        "name": "narrative_faithfulness",
        "type": "llm_judge",
        "description": "Is the report narrative faithful to the transcript (nothing added/omitted)?",
        "passing_threshold": 0.8,
        "judge_prompt": (
            "Compare the report narrative to the transcript. Penalize fabrications "
            "(content not stated by the officer) and material omissions. Return 0..1."
        ),
    },
    {
        "name": "documentation_standards",
        "type": "deterministic",
        "description": "Do populated fields use agency-standard phrasing (statutory codes, force-continuum terms, Miranda specificity, medical disposition)?",
        "passing_threshold": 0.75,
        "evaluation": "The metric officer corrections most directly improve.",
    },
    {
        "name": "cross_document_consistency",
        "type": "llm_judge",
        "description": "Do shared facts (time/location/names/charges) agree across documents?",
        "passing_threshold": 0.9,
        "judge_prompt": (
            "Given multiple reports from one incident, check that shared facts do not "
            "contradict each other. Return 0..1 and name any contradiction."
        ),
    },
    {
        "name": "question_coverage",
        "type": "llm_judge",
        "description": "Did the agent ask about required fields it did not already hear?",
        "passing_threshold": 0.6,
        "judge_prompt": (
            "Given the conversation and the report's empty priority-1 fields, did the "
            "agent ask about each gap? Return the fraction asked, 0..1."
        ),
    },
]


# --------------------------------------------------------------------------- #
# Corrections -> regression scenarios
# --------------------------------------------------------------------------- #
def corrections_to_scenarios(corrections: list[dict] | None = None) -> list[dict]:
    """Turn officer/teacher corrections into Cekura regression-scenario specs.

    Each correction asserts the agent, given a similar transcript, should produce
    the corrected phrasing for that field — a permanent guard against the mistake
    recurring (the "boss's list of past mistakes" from the briefing).
    """
    if corrections is None:
        from improvement.correction_store import correction_store

        corrections = correction_store.corrections

    scenarios = []
    for i, c in enumerate(corrections):
        if not c.get("transcript") or not c.get("field"):
            continue
        scenarios.append(
            {
                "name": f"regression_{c.get('report_type','report')}_{c['field']}_{i}",
                "tags": ["regression", "correction", c.get("report_type", "report")],
                "transcript": c["transcript"],
                "assertion": (
                    f"In the {c.get('report_type','report')}, field '{c['field']}' "
                    f"should follow the convention exemplified by "
                    f"\"{c.get('corrected')}\" (not \"{c.get('original')}\")."
                ),
                "category": c.get("category", "general"),
            }
        )
    return scenarios


def red_team_scenarios() -> list[dict]:
    """Adversarial cases targeting the Axon Draft One failure mode: the agent must
    use null rather than invent when the narrative is ambiguous or noisy."""
    return [
        {
            "name": "redteam_ambient_noise_frog",
            "tags": ["red_team", "hallucination"],
            "transcript": (
                "OFFICER: Uh, I responded to a call, there was a TV playing some "
                "cartoon in the background, something about a guy turning into a frog. "
                "Anyway the actual call was a noise complaint, nobody was home."
            ),
            "assertion": (
                "No report field may mention a frog, a transformation, or any cartoon "
                "content. suspect/charges/arrest fields must be null. incident_type ~ "
                "'noise complaint'."
            ),
            "category": "hallucination",
        },
        {
            "name": "redteam_unstated_charges",
            "tags": ["red_team", "hallucination"],
            "transcript": (
                "OFFICER: I took a report about a stolen bicycle from a front yard. "
                "No suspect, no witnesses, owner just wants it documented."
            ),
            "assertion": (
                "charges, arrestee_name, miranda_given must be null. The agent must "
                "not invent an arrest that did not happen."
            ),
            "category": "hallucination",
        },
        {
            "name": "redteam_missing_time",
            "tags": ["red_team", "completeness"],
            "transcript": (
                "OFFICER: Vandalism report, graffiti on the wall at the rec center. "
                "I'm not sure exactly when it happened, sometime this week."
            ),
            "assertion": "incident_time should be null (not fabricated to a precise time).",
            "category": "hallucination",
        },
    ]


# --------------------------------------------------------------------------- #
# Experiment -> Cekura call-log payloads
# --------------------------------------------------------------------------- #
def _batch_payload(label: str, evals: dict) -> list[dict]:
    """Shape one pass (baseline|improved) as Cekura-ingestable call logs."""
    out = []
    for sid, ev in evals.items():
        metrics = {
            "escalation_correctness": ev["escalation_correctness"]["score"],
            "question_coverage": ev["question_coverage"]["score"],
            "cross_document_consistency": ev["cross_document_consistency"]["score"],
            "overall_score": ev["overall_score"],
        }
        for rt, m in ev["per_report"].items():
            for mname, verdict in m.items():
                metrics[f"{rt}.{mname}"] = verdict["score"]
        out.append({"scenario_id": sid, "variant": label, "metrics": metrics})
    return out


def build_cekura_payloads(result: dict) -> dict:
    """Build the full Cekura payload bundle for a self-improvement run."""
    return {
        "metrics": METRIC_DEFINITIONS,
        "regression_scenarios": corrections_to_scenarios()
        if result.get("corrections_learned")
        else [],
        "red_team_scenarios": red_team_scenarios(),
        "baseline_batch": _batch_payload("baseline", result["baseline_eval"]),
        "improved_batch": _batch_payload("improved", result["improved_eval"]),
        "gate": result["gate"],
    }


def push_experiment(result: dict, out_dir: str | Path | None = None) -> Path:
    """Serialize the Cekura payloads next to the experiment artifacts.

    The MCP-driven creation of metrics/scenarios/runs/dashboard reads this file;
    keeping it on disk makes the dashboard reproducible from a run. (The live
    upload itself is performed via the Cekura MCP tools from Claude Code.)
    """
    payloads = build_cekura_payloads(result)
    if out_dir is None:
        base = Path(os.getenv("SAFELINE_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
        out_dir = base / "experiments"
        out_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(out_dir)
    path = out_dir / "cekura_payload.json"
    path.write_text(json.dumps(payloads, indent=2))
    logger.info(f"[cekura] payload written to {path} ({len(payloads['baseline_batch'])} scenarios)")
    return path


def fetch_run_scores(run_id) -> dict:
    """Best-effort fetch of a Cekura run's per-scenario scores for the hosted gate.

    Returns ``{scenario_id: evaluate-shaped dict}``. Without Cekura creds or the
    SDK this raises so callers can fall back to the local gate.
    """
    if not os.getenv("CEKURA_API_KEY"):
        raise RuntimeError("CEKURA_API_KEY not set; use the local regression_gate.check instead.")
    raise NotImplementedError(
        "Live run fetch is performed via the Cekura MCP tools (results_list / "
        "runs_bulk_retrieve) from Claude Code; wire those results into "
        "regression_gate.check()."
    )

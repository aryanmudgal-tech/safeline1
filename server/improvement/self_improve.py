"""Self-improvement harness — the closed loop, end to end (auto-improvement).

This is the experiment runner that PROVES the loop works. For each demo scenario
it:

  1. BASELINE   — generates reports with learning OFF (a clean, un-taught agent)
                  and scores them with the five evaluators.
  2. TEACH      — an autonomous teacher (``teacher.py``) writes the ideal report
                  and the diff vs. the baseline becomes corrections, fed into an
                  ISOLATED correction store (so the experiment is reproducible and
                  never touches the production corrections log).
  3. IMPROVED   — regenerates the same reports with learning ON (the compiled
                  guidance from step 2 is injected) and scores them again.
  4. DELTA+GATE — reports the per-metric, per-scenario before/after lift and runs
                  the Layer-2 regression gate (PASS only if nothing regressed).

Artifacts (baseline.json, improved.json, report.md) are written under
``data/experiments/<timestamp>/``.

Run it::

    cd server
    uv run python -m improvement.self_improve --demo          # full before/after
    uv run python -m improvement.self_improve --offline       # heuristic judges (free)
    uv run python -m improvement.self_improve --holdout simple_theft_incident_only
    uv run python -m improvement.self_improve --self-critique  # + reflexion pass

Note: report GENERATION always calls the configured LLM (the agent itself needs an
OPENAI_API_KEY). ``--offline`` only forces the *scoring* to use heuristics instead
of LLM judges, so the harness still runs (and the lift is still measured) with no
judge spend. With no API key at all the harness runs as a structural smoke test
(all fields null, zero delta).
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import sys
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parent.parent
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from loguru import logger  # noqa: E402

from cekura_eval import evaluate_incident_async  # noqa: E402
from improvement import teacher  # noqa: E402
from improvement.correction_store import CorrectionStore  # noqa: E402
from improvement.regression_gate import check as gate_check  # noqa: E402
from improvement.regression_gate import summary_line  # noqa: E402
from report_generator import generate_reports, load_template  # noqa: E402

_SCENARIOS_PATH = _SERVER_DIR / "mock_data" / "demo_scenarios.json"
_EXPERIMENTS_DIR = Path(os.getenv("SAFELINE_DATA_DIR", _SERVER_DIR / "data")) / "experiments"

_METRIC_LABELS = {
    "report_completeness": "Report Completeness",
    "extraction_accuracy": "Extraction Accuracy",
    "narrative_faithfulness": "Narrative Faithfulness",
    "documentation_standards": "Documentation Standards",
    "cross_document_consistency": "Cross-Doc Consistency",
    "question_coverage": "Question Coverage",
    "escalation_correctness": "Escalation Correctness",
}


def load_scenarios(scenario_ids: list[str] | None = None) -> list[dict]:
    with open(_SCENARIOS_PATH) as f:
        scenarios = json.load(f)
    if scenario_ids:
        wanted = set(scenario_ids)
        scenarios = [s for s in scenarios if s["id"] in wanted]
    return scenarios


def _transcript(scenario: dict) -> str:
    """The officer's narrative as a single-speaker transcript."""
    return f"OFFICER: {scenario['narrative']}"


def _build_record(scenario: dict, reports: dict) -> dict:
    """Wrap generate_reports output into the record shape the evaluators expect,
    without touching disk."""
    reports_block = {}
    for rt, fields in reports.items():
        try:
            template = load_template(rt)["required_fields"]
        except FileNotFoundError:
            template = [{"field": k, "label": k, "priority": 2} for k in fields if not k.startswith("_")]
        values = {k: v for k, v in fields.items() if not k.startswith("_")}
        reports_block[rt] = {
            "_report_type": rt,
            "template": template,
            "ai_draft": dict(values),
            "officer_version": dict(values),
        }
    return {
        "incident_id": scenario["id"],
        "transcript": _transcript(scenario),
        "is_emergency": bool(scenario.get("is_emergency")),
        "reports": reports_block,
    }


# --------------------------------------------------------------------------- #
# Passes
# --------------------------------------------------------------------------- #
async def _generate_and_eval(scenario, *, use_learning, store, self_critique, use_llm):
    transcript = _transcript(scenario)
    reports = await generate_reports(
        transcript,
        scenario["expected_reports"],
        scenario["badge_number"],
        use_learning=use_learning,
        self_critique=self_critique,
        store=store,
    )
    record = _build_record(scenario, reports)
    evaluation = await evaluate_incident_async(record, use_llm=use_llm)
    return record, evaluation


async def run_experiment(
    scenario_ids: list[str] | None = None,
    *,
    use_llm: bool = True,
    self_critique: bool = False,
    holdout: str | None = None,
) -> dict:
    scenarios = load_scenarios(scenario_ids)
    if not scenarios:
        raise SystemExit("No scenarios matched.")

    has_key = bool(os.getenv("OPENAI_API_KEY"))
    if not has_key:
        logger.warning(
            "OPENAI_API_KEY not set — running as a structural smoke test "
            "(reports will be empty and the delta will be ~0)."
        )

    # Isolated, empty store: the delta is attributable purely to what the teacher
    # learns in THIS run. No synthetic seeds, no production log, no persistence.
    store = CorrectionStore(persist=False, load_synthetic=False, load_logged=False)

    # 1. BASELINE -----------------------------------------------------------
    logger.info("[1/3] BASELINE pass (learning OFF)…")
    baseline_records, baseline_eval = {}, {}
    for s in scenarios:
        rec, ev = await _generate_and_eval(
            s, use_learning=False, store=store, self_critique=self_critique, use_llm=use_llm
        )
        baseline_records[s["id"]], baseline_eval[s["id"]] = rec, ev
        logger.info(f"  baseline {s['id']}: overall={ev['overall_score']:.3f}")

    # 2. TEACH --------------------------------------------------------------
    teach_set = [s for s in scenarios if s["id"] != holdout] if holdout else scenarios
    logger.info(f"[2/3] TEACH pass (autonomous corrections from {len(teach_set)} scenarios)…")
    corrections = []
    for s in teach_set:
        transcript = _transcript(s)
        for rt in s["expected_reports"]:
            try:
                template_fields = load_template(rt)["required_fields"]
            except FileNotFoundError:
                continue
            gold = await teacher.synthesize_gold(transcript, rt, template_fields)
            if not gold:
                continue
            ai_draft = baseline_records[s["id"]]["reports"].get(rt, {}).get("ai_draft", {})
            added = teacher.derive_corrections(
                transcript, rt, ai_draft, gold, store=store, persist=False
            )
            corrections.extend(added)
    logger.info(f"  learned {len(corrections)} corrections (store size={len(store)})")

    # 3. IMPROVED -----------------------------------------------------------
    logger.info("[3/3] IMPROVED pass (learning ON)…")
    improved_records, improved_eval = {}, {}
    for s in scenarios:
        rec, ev = await _generate_and_eval(
            s, use_learning=True, store=store, self_critique=self_critique, use_llm=use_llm
        )
        improved_records[s["id"]], improved_eval[s["id"]] = rec, ev
        logger.info(f"  improved {s['id']}: overall={ev['overall_score']:.3f}")

    # 4. GATE ---------------------------------------------------------------
    gate = gate_check(baseline_eval, improved_eval)

    # Raw transcripts + generated report values, so the Cekura push (and any
    # other consumer) can rebuild the actual call content, not just the scores.
    def _raw(records):
        out = {}
        for sid, rec in records.items():
            out[sid] = {
                rt: dict(r.get("ai_draft") or {}) for rt, r in rec["reports"].items()
            }
        return out

    transcripts = {s["id"]: _transcript(s) for s in scenarios}

    return {
        "generated_at": datetime.datetime.now().isoformat(),
        "scenarios": [s["id"] for s in scenarios],
        "holdout": holdout,
        "use_llm_judges": use_llm and has_key,
        "self_critique": self_critique,
        "corrections_learned": len(corrections),
        "baseline_eval": baseline_eval,
        "improved_eval": improved_eval,
        "transcripts": transcripts,
        "baseline_reports": _raw(baseline_records),
        "improved_reports": _raw(improved_records),
        "gate": gate,
        "_corrections_sample": [
            {k: c.get(k) for k in ("report_type", "field", "original", "corrected")}
            for c in corrections[:12]
        ],
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _metric_means(evals: dict) -> dict[str, float]:
    """Average each leaf metric across scenarios -> {label: mean}."""
    buckets: dict[str, list[float]] = {}

    def add(name, score):
        buckets.setdefault(name, []).append(score)

    for ev in evals.values():
        for key in ("escalation_correctness", "question_coverage", "cross_document_consistency"):
            add(_METRIC_LABELS[key], ev[key]["score"])
        for metrics in ev["per_report"].values():
            for mname, verdict in metrics.items():
                add(_METRIC_LABELS.get(mname, mname), verdict["score"])
    return {name: round(sum(v) / len(v), 3) for name, v in buckets.items()}


def render_markdown(result: dict) -> str:
    base_means = _metric_means(result["baseline_eval"])
    impr_means = _metric_means(result["improved_eval"])
    gate = result["gate"]

    lines = [
        "# Safeline Self-Improvement Report",
        "",
        f"- Generated: {result['generated_at']}",
        f"- Scenarios: {', '.join(result['scenarios'])}",
        f"- Holdout: {result['holdout'] or 'none (teach on all)'}",
        f"- Scoring engine: {'LLM judges' if result['use_llm_judges'] else 'heuristic'}",
        f"- Self-critique pass: {'on' if result['self_critique'] else 'off'}",
        f"- Corrections learned this run: {result['corrections_learned']}",
        "",
        f"## Verdict: {summary_line(gate)}",
        "",
        "## Per-metric lift (mean across scenarios)",
        "",
        "| Metric | Baseline | Improved | Δ |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in sorted(set(base_means) | set(impr_means)):
        b, i = base_means.get(name, 0.0), impr_means.get(name, 0.0)
        lines.append(f"| {name} | {b:.3f} | {i:.3f} | {i - b:+.3f} |")

    lines += [
        "",
        "## Per-scenario overall",
        "",
        "| Scenario | Baseline | Improved | Δ |",
        "| --- | ---: | ---: | ---: |",
    ]
    for sid in result["scenarios"]:
        b = result["baseline_eval"][sid]["overall_score"]
        i = result["improved_eval"][sid]["overall_score"]
        lines.append(f"| {sid} | {b:.3f} | {i:.3f} | {i - b:+.3f} |")

    if gate["regressions"]:
        lines += ["", "## ⛔ Regressions (blocked promotion)", ""]
        for r in gate["regressions"]:
            reason = f" — {r['reason']}" if r.get("reason") else ""
            lines.append(
                f"- {r['scenario']} · {r['metric']}: {r['baseline']:.3f} → "
                f"{r['candidate']:.3f} ({r['delta']:+.3f}){reason}"
            )

    if gate.get("soft_regressions"):
        lines += ["", "## ⚠️ Soft dips (allowed — net-positive scenario, flagged for review)", ""]
        for r in gate["soft_regressions"]:
            lines.append(f"- {r['scenario']} · {r['metric']}: {r['baseline']:.3f} → {r['candidate']:.3f} ({r['delta']:+.3f})")

    if result["_corrections_sample"]:
        lines += ["", "## Sample corrections the agent learned", ""]
        for c in result["_corrections_sample"]:
            lines.append(f"- `{c['report_type']}.{c['field']}`: \"{c['original']}\" → \"{c['corrected']}\"")

    return "\n".join(lines) + "\n"


def _print_console(result: dict):
    base_means = _metric_means(result["baseline_eval"])
    impr_means = _metric_means(result["improved_eval"])
    print("\n" + "=" * 64)
    print("  SAFELINE SELF-IMPROVEMENT LOOP — BEFORE / AFTER")
    print("=" * 64)
    print(f"{'Metric':<26}{'Baseline':>10}{'Improved':>10}{'Δ':>9}")
    print("-" * 64)
    for name in sorted(set(base_means) | set(impr_means)):
        b, i = base_means.get(name, 0.0), impr_means.get(name, 0.0)
        print(f"{name:<26}{b:>10.3f}{i:>10.3f}{i - b:>+9.3f}")
    print("-" * 64)
    print(f"  {summary_line(result['gate'])}")
    print("=" * 64 + "\n")


def _write_artifacts(result: dict) -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = _EXPERIMENTS_DIR / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "baseline.json").write_text(json.dumps(result["baseline_eval"], indent=2))
    (out_dir / "improved.json").write_text(json.dumps(result["improved_eval"], indent=2))
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    (out_dir / "report.md").write_text(render_markdown(result))
    return out_dir


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None):
    parser = argparse.ArgumentParser(description="Safeline self-improvement harness")
    parser.add_argument("--demo", action="store_true", help="Run all demo scenarios (default).")
    parser.add_argument("--scenarios", nargs="*", help="Specific scenario ids to run.")
    parser.add_argument("--offline", action="store_true", help="Score with heuristics, not LLM judges.")
    parser.add_argument("--self-critique", action="store_true", help="Add a reflexion pass during generation.")
    parser.add_argument("--holdout", help="Teach on all scenarios EXCEPT this id (held-out lift).")
    parser.add_argument("--report", help="Also copy the markdown report to this path.")
    parser.add_argument("--cekura", action="store_true", help="Push baseline/improved batches to Cekura.")
    args = parser.parse_args(argv)

    result = asyncio.run(
        run_experiment(
            scenario_ids=args.scenarios,
            use_llm=not args.offline,
            self_critique=args.self_critique,
            holdout=args.holdout,
        )
    )

    out_dir = _write_artifacts(result)
    _print_console(result)
    print(f"Artifacts written to: {out_dir}")
    print(f"  report.md  baseline.json  improved.json  result.json")

    if args.report:
        Path(args.report).write_text(render_markdown(result))
        print(f"Report also written to: {args.report}")

    if args.cekura:
        try:
            from improvement.cekura_sync import push_experiment

            push_experiment(result)
        except Exception as e:
            print(f"[cekura] push skipped: {e}")

    # Non-zero exit on a blocked promotion so this can gate CI / a deploy hook.
    raise SystemExit(0 if result["gate"]["promote"] else 1)


if __name__ == "__main__":
    main()

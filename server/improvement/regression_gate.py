"""Regression gate — auto-improvement Layer 2's "do no harm" check.

Layer 1 makes the agent better on average, but a prompt/guidance change can still
*regress* an individual scenario or metric. Layer 2's job is to block any update
that makes something worse. This is the analogy from the briefing: the boss does a
spot-check before a report goes out and keeps a list of past mistakes so old ones
can't creep back in.

``check`` compares per-scenario, per-metric scores between a baseline run and a
candidate run and returns a promote/block decision plus the exact regressions. The
harness uses it to label the experiment PASS/BLOCK; a deploy hook can use the same
function to gate a prompt change before it ships.

``check_via_cekura`` is the hosted equivalent: it compares two Cekura runs by id so
the same gate can run against dashboard scores (wired by ``cekura_sync``).
"""

from __future__ import annotations

# A per-metric drop smaller than this is treated as noise (LLM judges vary run to
# run, and move in coarse steps). Drops beyond it are scrutinized by the policy.
DEFAULT_TOLERANCE = 0.05
# A single metric falling by more than this is a "cliff" and blocks promotion on its
# own, regardless of how much the scenario improved elsewhere.
DEFAULT_HARD_FLOOR = 0.15


def _leaf_scores(scenario_result: dict) -> dict[str, float]:
    """Flatten one scenario's evaluation into ``{metric_path: score}``."""
    scores: dict[str, float] = {}
    for key in ("escalation_correctness", "question_coverage", "cross_document_consistency"):
        if key in scenario_result:
            scores[key] = scenario_result[key]["score"]
    for report_type, metrics in scenario_result.get("per_report", {}).items():
        for metric_name, verdict in metrics.items():
            scores[f"{report_type}.{metric_name}"] = verdict["score"]
    return scores


def check(
    baseline: dict[str, dict],
    candidate: dict[str, dict],
    tolerance: float = DEFAULT_TOLERANCE,
    hard_floor: float = DEFAULT_HARD_FLOOR,
) -> dict:
    """Compare baseline vs candidate evaluations keyed by scenario id.

    Two-tier policy (robust against LLM-judge noise, strict against real harm):
      * A metric drop beyond ``hard_floor`` is a CLIFF — it blocks on its own.
      * A metric drop beyond ``tolerance`` blocks only if that scenario ALSO got
        worse overall (i.e. it's not a net-positive tradeoff).
      * Smaller dips on scenarios that improved overall are reported as
        ``soft_regressions`` (flagged for human review) but do not block.

    Args:
        baseline / candidate: ``{scenario_id: evaluate_incident(...) result}``.

    Returns ``{promote, regressions, soft_regressions, improvements,
    baseline_overall, candidate_overall, delta}``.
    """
    regressions, soft_regressions, improvements = [], [], []
    base_overall, cand_overall = [], []

    for scenario_id, base_res in baseline.items():
        cand_res = candidate.get(scenario_id)
        if not cand_res:
            continue
        s_base = base_res.get("overall_score", 0.0)
        s_cand = cand_res.get("overall_score", 0.0)
        base_overall.append(s_base)
        cand_overall.append(s_cand)
        scenario_improved = s_cand >= s_base

        base_leaf = _leaf_scores(base_res)
        cand_leaf = _leaf_scores(cand_res)
        for metric, base_score in base_leaf.items():
            cand_score = cand_leaf.get(metric)
            if cand_score is None:
                continue
            delta = round(cand_score - base_score, 3)
            entry = {
                "scenario": scenario_id,
                "metric": metric,
                "baseline": base_score,
                "candidate": cand_score,
                "delta": delta,
            }
            if delta > tolerance:
                improvements.append(entry)
            elif delta < -hard_floor:
                entry["reason"] = "cliff"
                regressions.append(entry)
            elif delta < -tolerance:
                if scenario_improved:
                    soft_regressions.append(entry)  # net-positive tradeoff, allowed
                else:
                    entry["reason"] = "scenario regressed overall"
                    regressions.append(entry)

    base_mean = round(sum(base_overall) / len(base_overall), 3) if base_overall else 0.0
    cand_mean = round(sum(cand_overall) / len(cand_overall), 3) if cand_overall else 0.0

    return {
        "promote": len(regressions) == 0,
        "tolerance": tolerance,
        "hard_floor": hard_floor,
        "regressions": regressions,
        "soft_regressions": soft_regressions,
        "improvements": improvements,
        "baseline_overall": base_mean,
        "candidate_overall": cand_mean,
        "delta": round(cand_mean - base_mean, 3),
    }


def summary_line(gate: dict) -> str:
    """One-line verdict for logs/CLI."""
    verdict = "PASS ✅ promote" if gate["promote"] else "BLOCK ⛔ regressions found"
    soft = len(gate.get("soft_regressions", []))
    soft_str = f", {soft} soft-dip" if soft else ""
    return (
        f"{verdict} | overall {gate['baseline_overall']:.3f} -> "
        f"{gate['candidate_overall']:.3f} (Δ{gate['delta']:+.3f}) | "
        f"{len(gate['improvements'])} improved, {len(gate['regressions'])} regressed{soft_str}"
    )


def check_via_cekura(baseline_run_id, candidate_run_id, tolerance: float = DEFAULT_TOLERANCE) -> dict:
    """Hosted variant: gate two Cekura runs by id.

    Implemented by ``improvement.cekura_sync`` (which has the MCP-backed run
    fetchers); kept here as the canonical entry point so callers import the gate
    from one place.
    """
    from improvement.cekura_sync import fetch_run_scores

    baseline = fetch_run_scores(baseline_run_id)
    candidate = fetch_run_scores(candidate_run_id)
    return check(baseline, candidate, tolerance=tolerance)

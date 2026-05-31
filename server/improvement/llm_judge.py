"""LLM-judge evaluators for Safeline reports (auto-improvement Layer 2).

These are the "smart" versions of the evaluators in ``cekura_eval.py``. Where the
heuristics in that module score with token-overlap arithmetic, the judges here ask
an LLM to actually read the transcript and the generated report and reason about
faithfulness / hallucination / consistency — the same job a Cekura LLM-judge metric
does in the dashboard.

Design contract: **a judge must never break a run.** If there is no API key, the
network is down, or the model returns garbage, every judge transparently falls back
to the corresponding deterministic heuristic in ``cekura_eval`` and tags the result
with ``"engine": "heuristic"``. So ``self_improve.py --offline`` and CI work with no
network, while a fully-configured machine gets real judge scores.

Each judge returns a dict shaped like the heuristics::

    {"score": float 0..1, "passed": bool, "detail": str,
     "reasoning": str, "engine": "llm" | "heuristic"}

Public API:
    await judge_extraction_accuracy(report, transcript)
    await judge_narrative_faithfulness(report, transcript)
    await judge_cross_document_consistency(reports)
"""

from __future__ import annotations

import json
import os

from loguru import logger

# --------------------------------------------------------------------------- #
# Config / lazy client
# --------------------------------------------------------------------------- #
_client = None


def _judge_model() -> str:
    # A small, cheap model is plenty for scoring; override with SAFELINE_JUDGE_MODEL.
    return os.getenv("SAFELINE_JUDGE_MODEL", os.getenv("REPORT_GEN_MODEL", "gpt-4o-mini"))


def judges_available() -> bool:
    """True when an LLM judge can actually run (an API key is configured)."""
    return bool(os.getenv("OPENAI_API_KEY"))


def _get_client():
    global _client
    if _client is None:
        from openai import AsyncOpenAI

        _client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


# A tiny per-process cache keyed by (judge, prompt) so re-scoring the same
# report inside one experiment run doesn't pay the API twice.
_cache: dict[tuple[str, str], dict] = {}


async def _call_judge(kind: str, prompt: str) -> dict | None:
    """Call the judge model and parse its JSON verdict. Return None on any failure
    so the caller can fall back to the heuristic."""
    if not judges_available():
        return None

    key = (kind, prompt)
    if key in _cache:
        return _cache[key]

    try:
        resp = await _get_client().chat.completions.create(
            model=_judge_model(),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a meticulous evaluator of police report quality. "
                        "You return ONLY a JSON object with keys: score (a number "
                        "from 0 to 1), passed (boolean), reasoning (one short "
                        "sentence). Be strict: hallucinations and unsupported "
                        "claims must lower the score sharply."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        data = json.loads(resp.choices[0].message.content)
        score = float(data.get("score", 0.0))
        score = max(0.0, min(1.0, score))
        verdict = {
            "score": round(score, 3),
            "passed": bool(data.get("passed", score >= 0.8)),
            "reasoning": str(data.get("reasoning", ""))[:400],
            "engine": "llm",
        }
        _cache[key] = verdict
        return verdict
    except Exception as e:  # no key, network, parse, rate limit, ...
        logger.warning(f"LLM judge '{kind}' failed ({e}); falling back to heuristic")
        return None


def _populated(report: dict) -> dict:
    version = report.get("officer_version") or report.get("ai_draft", {})
    return {k: v for k, v in version.items() if v not in (None, "", [])}


# --------------------------------------------------------------------------- #
# Judges (each falls back to the matching heuristic in cekura_eval)
# --------------------------------------------------------------------------- #
async def judge_extraction_accuracy(report: dict, transcript: str) -> dict:
    """Is every populated value actually grounded in the transcript (no
    hallucinations)? This is the "shapeshifted into a frog" guard."""
    from cekura_eval import extraction_accuracy  # heuristic fallback

    values = _populated(report)
    if not values:
        return {**extraction_accuracy(report, transcript), "engine": "heuristic"}

    prompt = (
        "Below is a transcript of a police officer's account and a set of fields a "
        "system extracted from it. Score how well the extracted values are GROUNDED "
        "in the transcript. A value is grounded only if the transcript supports it. "
        "Any invented, assumed, or contradicted value should sharply reduce the "
        "score. Do not reward values that merely sound plausible.\n\n"
        "IMPORTANT: Coding a fact the officer DID state into correct standard "
        "terminology is NOT a hallucination and must NOT be penalized. Examples of "
        "legitimate standardization: 'DUI' -> a statutory code like 'VC 23152(a)'; "
        "'physical force'/'took him down' -> a force-continuum term like 'empty-hand "
        "control (takedown)'; 'read him his rights' -> 'Miranda advised prior to "
        "questioning'. Only penalize values whose underlying FACT is absent from or "
        "contradicted by the transcript.\n\n"
        f"TRANSCRIPT:\n{transcript}\n\n"
        f"EXTRACTED FIELDS (JSON):\n{json.dumps(values, indent=2)}\n\n"
        "Return JSON: score (fraction of fields with a grounded underlying fact, "
        "0..1), passed (true if score >= 0.8), reasoning."
    )
    verdict = await _call_judge("extraction_accuracy", prompt)
    if verdict is None:
        return {**extraction_accuracy(report, transcript), "engine": "heuristic"}
    verdict.setdefault("detail", "LLM-judged grounding of extracted values")
    return verdict


async def judge_narrative_faithfulness(report: dict, transcript: str) -> dict:
    """Is the report's narrative faithful to the transcript — nothing added that
    the officer didn't say, nothing material omitted?"""
    from cekura_eval import narrative_faithfulness  # heuristic fallback

    version = report.get("officer_version") or report.get("ai_draft", {})
    narrative = str(version.get("narrative") or "").strip()
    if not narrative:
        return {**narrative_faithfulness(report, transcript), "engine": "heuristic"}

    prompt = (
        "Compare the report NARRATIVE against the officer's TRANSCRIPT. Penalize "
        "anything in the narrative that was NOT stated by the officer (fabrication) "
        "and any material fact from the transcript that was omitted.\n\n"
        f"TRANSCRIPT:\n{transcript}\n\n"
        f"NARRATIVE:\n{narrative}\n\n"
        "Return JSON: score (0..1, faithfulness), passed (score >= 0.8), reasoning."
    )
    verdict = await _call_judge("narrative_faithfulness", prompt)
    if verdict is None:
        return {**narrative_faithfulness(report, transcript), "engine": "heuristic"}
    verdict.setdefault("detail", "LLM-judged narrative faithfulness")
    return verdict


async def judge_cross_document_consistency(reports: dict) -> dict:
    """Do shared facts (times, names, locations, charges) agree across the
    generated documents for one incident?"""
    from cekura_eval import cross_document_consistency  # heuristic fallback

    if len(reports) < 2:
        return {**cross_document_consistency(reports), "engine": "heuristic"}

    flat = {
        rt: (r.get("officer_version") or r.get("ai_draft", {}))
        for rt, r in reports.items()
    }
    prompt = (
        "These are multiple police reports generated from ONE incident. Check that "
        "facts shared across documents are consistent: dates, times, locations, "
        "person names, and charges must not contradict each other.\n\n"
        f"REPORTS (JSON):\n{json.dumps(flat, indent=2)}\n\n"
        "Return JSON: score (0..1, 1 = fully consistent), passed (score >= 0.9), "
        "reasoning naming any contradiction."
    )
    verdict = await _call_judge("cross_document_consistency", prompt)
    if verdict is None:
        return {**cross_document_consistency(reports), "engine": "heuristic"}
    verdict.setdefault("detail", "LLM-judged cross-document consistency")
    return verdict

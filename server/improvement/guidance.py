"""Compiled learned guidance — the robust core of auto-improvement Layer 1.

The naive version of Layer 1 dumped raw past corrections into the prompt as
free-text few-shot examples. That works, but it is noisy (it can leak the *values*
from another incident), token-hungry, and the retrieved set jitters between runs,
so the measured before/after delta is unstable.

This module compiles corrections into a small, deterministic **field rubric**
instead. Corrections are grouped by ``(report_type, field)`` and turned into one
concise instruction per field — e.g.::

    - charges: cite the statutory code (e.g. "VC 23152(a)"), not just the
      plain-language offense.
    - force_type: use agency force-continuum terminology.

This is essentially what DSPy's MIPROv2 would converge toward (optimized
instructions), but produced deterministically so it is reproducible and cheap.
Retrieval still matters: rubric lines whose source transcripts resemble the
current incident are ranked first, so the guidance stays relevant.

Public API:
    build_guidance_block(transcript, report_type, store=None, max_lines=6) -> str
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _instruction_for(field: str, examples: list[dict]) -> str:
    """Render one rubric line for a field from its accumulated corrections.

    We surface the most recent corrected phrasing as the canonical convention and
    fold in the officer's stated reason (``note``) when present, but we never copy
    the literal *value* as a fact to assert — it's framed as a phrasing/style
    convention so the model applies the convention, not the other incident's data.
    """
    # A concrete phrasing contrast is the most actionable guidance; fall back to an
    # explicit note (the "why") only when no contrast is available.
    notes = [e["note"] for e in examples if e.get("note")]
    corrected = next((e.get("corrected") for e in examples if e.get("corrected")), None)
    original = next((e.get("original") for e in examples if e.get("original")), None)

    if corrected and original:
        guidance = (
            f'when this field applies, document it in the style of "{corrected}" '
            f'rather than "{original}"'
        )
    elif corrected:
        guidance = f'when this field applies, document it in the style of "{corrected}"'
    elif notes:
        guidance = notes[0].rstrip(".")
    else:
        guidance = "follow the department's standard phrasing for this field"

    return f"- {field}: {guidance}."


def build_guidance_block(
    transcript: str,
    report_type: str,
    store=None,
    max_lines: int = 6,
) -> str:
    """Build the learned-guidance block injected into the extraction prompt.

    Returns ``""`` when there is nothing learned yet (cold start), so the prompt
    is unchanged from baseline. ``store`` defaults to the shared singleton; the
    self-improvement harness passes an isolated store for clean experiments.
    """
    if store is None:
        try:
            from improvement.correction_store import correction_store as store

            # Pick up corrections approved by the (separate) review-portal process
            # since this process started — no bot restart needed. No-op for the
            # harness, which passes its own isolated store.
            store.refresh_from_disk()
        except Exception:
            return ""

    corrections = getattr(store, "corrections", None)
    if not corrections:
        return ""

    q_tokens = _tokenize(transcript)

    # Group corrections for this report type by field.
    by_field: dict[str, list[dict]] = {}
    field_relevance: dict[str, float] = {}
    for c in corrections:
        if report_type and c.get("report_type") and c.get("report_type") != report_type:
            continue
        field = c.get("field") or "general"
        by_field.setdefault(field, []).append(c)
        rel = _jaccard(q_tokens, _tokenize(c.get("transcript", "")))
        field_relevance[field] = max(field_relevance.get(field, 0.0), rel)

    if not by_field:
        return ""

    # Rank fields: transcript relevance first, then how often the field is
    # corrected (a field officers fix a lot is worth guiding on regardless).
    ranked_fields = sorted(
        by_field.keys(),
        key=lambda f: (field_relevance.get(f, 0.0), len(by_field[f])),
        reverse=True,
    )[:max_lines]

    lines = [
        "",
        "LEARNED CONVENTIONS — when a field below applies to THIS incident, document",
        "it using the professional standard shown. Coding a fact the officer DID state",
        "into its standard form (e.g. a statutory code for a stated offense, a force-",
        "continuum term for a described action) is expected and is NOT inventing. Never",
        "assert a fact the officer did not state:",
    ]
    for field in ranked_fields:
        lines.append(_instruction_for(field, by_field[field]))
    return "\n".join(lines) + "\n"

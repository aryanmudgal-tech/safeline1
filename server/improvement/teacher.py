"""Autonomous correction teacher (closes the self-improvement loop without a human).

In production the correction signal comes from officers editing reports in the
review portal. For a reproducible, self-driving demo we need that signal without a
human in the loop, so this module plays the role of an expert reviewer:

  1. ``synthesize_gold`` asks a strong "teacher" model to write the *ideal* report
     for a scenario — applying the department conventions officers care about
     (force-continuum terminology, statutory charge codes, Miranda phrasing,
     EMS/medical disposition, theft classification by statutory threshold).
  2. ``derive_corrections`` diffs the agent's baseline draft against that gold and
     emits one correction per materially-different field, which is fed into the
     correction store exactly as an officer edit would be.

Those corrections are what Layer 1 then compiles into guidance, so the "improved"
pass measurably closes the gap to the gold. The teacher is the *source* of the
learning signal; it is never used to score (that's the evaluators' job), so there
is no circularity in the before/after measurement.
"""

from __future__ import annotations

import json
import os

from loguru import logger

# Conventions the teacher enforces — the same ones in synthetic_corrections.json,
# generalized. Kept explicit so the teacher's "expertise" is auditable.
_CONVENTIONS = """Apply these professional law-enforcement documentation conventions:
- Charges: cite the statutory code section (e.g. "VC 23152(a) - DUI", "PC 243(e)(1) -
  Domestic Battery"), not only the plain-language offense.
- Force used: use agency force-continuum terminology (e.g. "Empty-hand control
  (takedown / control hold)") rather than vague phrases like "physical force".
- Miranda: state whether it was advised prior to custodial interrogation and where
  (e.g. "Yes - Miranda advised verbally at scene prior to questioning").
- Medical attention: capture EMS response, on-scene evaluation, and transport/refusal
  disposition (e.g. "EMS responded, subject evaluated on scene, refused transport").
- Theft: classify by statutory threshold (petty vs grand) using the stated loss value.
- Times/locations/names: use exactly what the officer stated; never invent."""


def _teacher_model() -> str:
    return os.getenv("SAFELINE_TEACHER_MODEL", os.getenv("REPORT_GEN_MODEL", "gpt-4o-mini"))


async def synthesize_gold(
    transcript: str,
    report_type: str,
    template_fields: list[dict],
    client=None,
) -> dict:
    """Produce the ideal (gold) report for one report type from the transcript.

    Returns ``{field: value|null}``. Falls back to ``{}`` on any error so the
    harness can skip teaching for that report rather than crash.
    """
    if not os.getenv("OPENAI_API_KEY"):
        return {}

    if client is None:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    fields = [f["field"] for f in template_fields]
    field_help = "\n".join(f'- {f["field"]}: {f["label"]}' for f in template_fields)

    prompt = f"""You are an expert police records supervisor writing a flawless \
{report_type.replace('_', ' ')} from an officer's verbal account.

{_CONVENTIONS}

Use null for any field the officer did not provide. Do NOT invent facts.

TRANSCRIPT:
{transcript}

Fields:
{field_help}

Return ONLY a JSON object with these exact field names."""
    try:
        resp = await client.chat.completions.create(
            model=_teacher_model(),
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        gold = json.loads(resp.choices[0].message.content)
        return {f: gold.get(f, None) for f in fields}
    except Exception as e:
        logger.warning(f"teacher.synthesize_gold failed for {report_type}: {e}")
        return {}


# Free-text prose fields are NOT standardizable conventions — teaching the agent to
# rewrite them toward another incident's phrasing causes narrative drift and hurts
# faithfulness. Layer 1 learns structured/standardizable fields only; the narrative
# itself is governed by the anti-hallucination rules in report_generator.
_PROSE_FIELDS = {
    "narrative",
    "incident_narrative",
    "statement",
    # Free-text justifications: keep them officer-grounded, don't restyle them.
    "force_justification",
    "subject_behavior",
}
_MAX_VALUE_LEN = 200  # corrected values longer than this are prose, not a convention


def _materially_different(field: str, original, corrected) -> bool:
    """True if the gold value is a meaningful, standardizable improvement.

    We only learn from fields the gold actually populated (corrected is non-empty),
    that differ from the draft, and that are short structured values — never the
    free-text narrative — so an empty gold field never teaches the agent to blank a
    field it correctly filled, and prose is never rewritten by the loop.
    """
    if field in _PROSE_FIELDS:
        return False
    if corrected in (None, "", []):
        return False
    if len(str(corrected)) > _MAX_VALUE_LEN:
        return False
    if (original or None) == (corrected or None):
        return False
    return str(original).strip().lower() != str(corrected).strip().lower()


def derive_corrections(
    transcript: str,
    report_type: str,
    ai_draft: dict,
    gold: dict,
    store=None,
    persist: bool = False,
) -> list[dict]:
    """Diff the baseline draft against the gold report and feed each material
    difference into the correction store as a learning example.

    ``persist`` is False by default so harness runs don't pollute the real
    on-disk corrections log; the live officer-edit path persists separately.
    Returns the list of correction records that were added.
    """
    if store is None:
        from improvement.correction_store import correction_store as store

    added = []
    for field, corrected in gold.items():
        original = ai_draft.get(field)
        if not _materially_different(field, original, corrected):
            continue
        rec = store.add_correction(
            transcript=transcript,
            original=original,
            corrected=corrected,
            category=f"{report_type}:{field}",
            report_type=report_type,
            field=field,
            note=None,  # let guidance use the concrete phrasing contrast
            _persist=persist,
            _source="teacher",
        )
        if rec is not None:  # None means it was a duplicate and was skipped
            added.append(rec)
    return added

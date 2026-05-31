"""Layer 3 — NVIDIA data flywheel: real correction → training-data export.

Layer 3 is the slow loop: once enough corrections accumulate, they are distilled
into the model's weights via LoRA fine-tuning (NeMo Customizer), so the agent stops
needing the Layer-1 notebook for common patterns — the prompt gets shorter and
latency drops. We cannot run a live fine-tune in a hackathon, but the *data* half
is real and runnable: this module converts accumulated corrections into a
fine-tuning-ready JSONL.

Architecture (what the full flywheel looks like):

    corrections_log.jsonl
        │  NeMo Curator   (dedupe, filter, balance by report_type/field)
        ▼
    training.jsonl (prompt → corrected completion)   ◀── export_training_jsonl()
        │  NeMo Customizer (LoRA on Nemotron 3 Nano)
        ▼
    candidate model
        │  NeMo Evaluator (run the Cekura suite: must beat base, no regressions)
        ▼
    promote → base model  →  retire internalized Layer-1 guidance  →  shorter prompt

Each exported example teaches the model to map the officer's transcript + the field
being filled to the officer-corrected value, in OpenAI/NeMo chat-tuning format.

Run it::

    uv run python -m improvement.finetune_export --out data/finetune/training.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_SERVER_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_OUT = _SERVER_DIR / "data" / "finetune" / "training.jsonl"

_SYSTEM = (
    "You are a police documentation assistant. Given an officer's narrative and a "
    "single field to populate, return the value using correct, agency-standard "
    "phrasing, or null if the narrative does not support it."
)


def _example(correction: dict) -> dict | None:
    """One chat-format tuning example from a correction (the corrected value is the
    target completion)."""
    field = correction.get("field")
    corrected = correction.get("corrected")
    transcript = correction.get("transcript")
    if not field or corrected in (None, "", []) or not transcript:
        return None
    report_type = (correction.get("report_type") or "report").replace("_", " ")
    user = (
        f"Report type: {report_type}\n"
        f"Field: {field}\n"
        f"Officer narrative:\n{transcript}\n\n"
        f"Provide the value for '{field}'."
    )
    return {
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": str(corrected)},
        ],
        "metadata": {
            "report_type": correction.get("report_type"),
            "field": field,
            "category": correction.get("category", "general"),
            "source": correction.get("_source", "officer"),
        },
    }


def _curate(corrections: list[dict]) -> list[dict]:
    """Stand-in for NeMo Curator: drop empties and de-duplicate identical
    (field, corrected, transcript) triples."""
    seen, curated = set(), []
    for c in corrections:
        key = (c.get("field"), str(c.get("corrected")), c.get("transcript"))
        if None in key[:2] or key in seen:
            continue
        seen.add(key)
        curated.append(c)
    return curated


def export_training_jsonl(out_path: str | Path = _DEFAULT_OUT, corrections=None) -> dict:
    """Write a fine-tuning-ready JSONL from accumulated corrections.

    Returns a small manifest ``{path, examples, by_report_type}``.
    """
    if corrections is None:
        from improvement.correction_store import correction_store

        corrections = correction_store.corrections

    curated = _curate(corrections)
    examples = [e for e in (_example(c) for c in curated) if e]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    by_type: dict[str, int] = {}
    for ex in examples:
        rt = ex["metadata"].get("report_type") or "unknown"
        by_type[rt] = by_type.get(rt, 0) + 1

    return {"path": str(out_path), "examples": len(examples), "by_report_type": by_type}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export corrections as fine-tuning JSONL (Layer 3).")
    parser.add_argument("--out", default=str(_DEFAULT_OUT), help="Output JSONL path.")
    args = parser.parse_args(argv)
    manifest = export_training_jsonl(args.out)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

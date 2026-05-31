"""Auto-improvement layer for Safeline (three layers).

Layer 1 — IMMEDIATE. A semantic correction store (``correction_store``) captures
every officer edit; ``guidance`` compiles those corrections into a concise field
rubric that is injected into the next report's extraction prompt. The very next
similar incident benefits.

Layer 2 — QUALITY GATE. Five evaluators (``cekura_eval`` + ``llm_judge``) score
every report; ``regression_gate`` blocks any prompt/guidance update that makes a
metric worse; ``cekura_sync`` mirrors metrics/scenarios/runs to the Cekura
dashboard and turns corrections into permanent regression tests.

Layer 3 — WEIGHTS. ``finetune_export`` distills accumulated corrections into a
LoRA fine-tuning dataset (NVIDIA NeMo data flywheel; architecture + real export).

The whole loop is exercised end to end by ``self_improve`` (baseline → teach →
improved → delta → gate). This is what Axon Draft One does NOT do: it deletes the
draft after export and never learns from the officer's edits.
"""

from .correction_store import CorrectionStore, correction_store
from . import guidance, regression_gate

__all__ = [
    "CorrectionStore",
    "correction_store",
    "guidance",
    "regression_gate",
]

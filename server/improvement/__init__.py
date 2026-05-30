"""Auto-improvement layer for Safeline.

Layer 1 (this package): a semantic correction store. Every time an officer
edits an AI-generated report and approves it, the diff is captured as a
"correction" and embedded into a vector index. When future reports are
generated, the most similar past corrections are injected into the extraction
prompt as few-shot examples, so the agent stops repeating the same mistakes.

This is what Axon Draft One does NOT do: it deletes the draft after export and
never learns from the officer's edits.
"""

from .correction_store import CorrectionStore, correction_store

__all__ = ["CorrectionStore", "correction_store"]

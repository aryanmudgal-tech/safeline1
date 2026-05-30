"""Semantic store of officer corrections (auto-improvement Layer 1).

Design goals for the hackathon:
  * Works out of the box. If ``sentence-transformers`` + ``faiss`` are installed
    we use real dense-vector retrieval (all-MiniLM-L6-v2, cosine via inner
    product on normalized vectors). If they're NOT installed, we transparently
    fall back to a lightweight pure-Python token-overlap similarity so the bot
    and report generator never crash on a fresh machine.
  * Corrections persist to disk (``mock_data/corrections_log.jsonl``) so the
    audit trail / learning loop survives process restarts — the opposite of
    Axon Draft One, which deletes the draft after export.

Public API:
    correction_store.add_correction(transcript, original, corrected, category, ...)
    correction_store.get_similar(transcript, k=3, report_type=None) -> list[dict]
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
from pathlib import Path

from loguru import logger

_HERE = Path(__file__).resolve().parent
_SERVER_DIR = _HERE.parent
_SYNTHETIC_PATH = _SERVER_DIR / "mock_data" / "synthetic_corrections.json"
_LOG_PATH = _SERVER_DIR / "mock_data" / "corrections_log.jsonl"

_EMBED_DIM = 384  # all-MiniLM-L6-v2

# --- Optional heavy backend (sentence-transformers + faiss) -----------------
_model = None
_faiss = None
_np = None


def _try_load_dense_backend():
    """Attempt to load the dense-embedding backend; return True on success."""
    global _model, _faiss, _np
    if _model is not None:
        return True
    try:
        import faiss  # type: ignore
        import numpy as np  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore

        logger.info("CorrectionStore: loading all-MiniLM-L6-v2 (dense retrieval)")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        _faiss = faiss
        _np = np
        return True
    except Exception as e:  # ImportError or model download failure
        logger.warning(
            f"CorrectionStore: dense backend unavailable ({e}); "
            "falling back to token-overlap similarity."
        )
        return False


# --- Lightweight pure-Python fallback ---------------------------------------
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


class CorrectionStore:
    """Stores officer corrections and retrieves semantically similar ones."""

    def __init__(self, persist: bool = True):
        self._lock = threading.Lock()
        self._persist = persist
        self.corrections: list[dict] = []
        self._dense = _try_load_dense_backend()
        self._index = None
        if self._dense:
            self._index = _faiss.IndexFlatIP(_EMBED_DIM)
        self._token_cache: list[set[str]] = []
        self._load_synthetic()
        self._load_logged()

    # -- embedding helpers ---------------------------------------------------
    def _embed(self, text: str):
        emb = _model.encode(text)
        emb = emb / (_np.linalg.norm(emb) + 1e-9)
        return emb.reshape(1, -1).astype("float32")

    # -- seeding -------------------------------------------------------------
    def _load_synthetic(self):
        try:
            with open(_SYNTHETIC_PATH) as f:
                seed = json.load(f)
        except FileNotFoundError:
            return
        for c in seed:
            self.add_correction(
                transcript=c.get("transcript", ""),
                original=c.get("original"),
                corrected=c.get("corrected"),
                category=c.get("category", "general"),
                report_type=c.get("report_type"),
                field=c.get("field"),
                note=c.get("note"),
                _persist=False,
                _source="synthetic",
            )

    def _load_logged(self):
        if not _LOG_PATH.exists():
            return
        try:
            with open(_LOG_PATH) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    c = json.loads(line)
                    self.add_correction(
                        transcript=c.get("transcript", ""),
                        original=c.get("original"),
                        corrected=c.get("corrected"),
                        category=c.get("category", "general"),
                        report_type=c.get("report_type"),
                        field=c.get("field"),
                        note=c.get("note"),
                        _persist=False,
                        _source=c.get("_source", "officer"),
                    )
        except Exception as e:
            logger.warning(f"CorrectionStore: failed to load logged corrections: {e}")

    # -- write ---------------------------------------------------------------
    def add_correction(
        self,
        transcript: str,
        original,
        corrected,
        category: str = "general",
        report_type: str | None = None,
        field: str | None = None,
        note: str | None = None,
        _persist: bool | None = None,
        _source: str = "officer",
    ):
        """Add a correction. ``transcript`` is the snippet of officer speech that
        produced ``original``; ``corrected`` is the officer's edited value."""
        record = {
            "transcript": transcript,
            "original": original,
            "corrected": corrected,
            "category": category,
            "report_type": report_type,
            "field": field,
            "note": note,
            "_source": _source,
        }
        with self._lock:
            if self._dense:
                self._index.add(self._embed(transcript))
            self._token_cache.append(_tokenize(f"{transcript} {field or ''}"))
            self.corrections.append(record)

        should_persist = self._persist if _persist is None else _persist
        if should_persist:
            self._append_log(record)
        return record

    def _append_log(self, record: dict):
        try:
            _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_LOG_PATH, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.warning(f"CorrectionStore: failed to persist correction: {e}")

    # -- read ----------------------------------------------------------------
    def get_similar(
        self, transcript: str, k: int = 3, report_type: str | None = None
    ) -> list[dict]:
        """Return up to ``k`` corrections most similar to ``transcript``.

        If ``report_type`` is given, corrections for that report type are
        preferred (they're the most relevant few-shot examples)."""
        with self._lock:
            if not self.corrections:
                return []
            n = len(self.corrections)

            if self._dense and self._index.ntotal == n:
                scores, indices = self._index.search(self._embed(transcript), min(k * 3, n))
                ranked = [
                    (self.corrections[i], float(s))
                    for s, i in zip(scores[0], indices[0])
                    if 0 <= i < n
                ]
            else:
                q = _tokenize(transcript)
                ranked = sorted(
                    ((self.corrections[i], _jaccard(q, self._token_cache[i])) for i in range(n)),
                    key=lambda x: x[1],
                    reverse=True,
                )

        # Prefer same report_type, then highest similarity.
        def sort_key(item):
            rec, score = item
            same_type = 1 if (report_type and rec.get("report_type") == report_type) else 0
            return (same_type, score)

        ranked.sort(key=sort_key, reverse=True)
        results = [rec for rec, score in ranked if score > 0][:k]
        return results

    def __len__(self) -> int:
        return len(self.corrections)


def _build_default_store() -> CorrectionStore:
    persist = os.getenv("SAFELINE_PERSIST_CORRECTIONS", "true").lower() != "false"
    return CorrectionStore(persist=persist)


# Module-level singleton used by report_generator and the review API.
correction_store = _build_default_store()

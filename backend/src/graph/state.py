"""
GraphState – the shared memory object that flows through every LangGraph node.

Key design choices
------------------
* `TypedDict` ensures static-analysis compatibility with LangGraph.
* All mutable containers default to empty lists (never ``None``).
* Pointers (`batch_index`, `current_page_start`) travel with the state so
  every node can be a pure function; no global mutable state is needed.
"""
from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Individual QA pair representation (in-flight / pre-commit)
# ---------------------------------------------------------------------------

class QAPair(TypedDict):
    """A single question-answer pair produced by the generator LLM."""
    question: str
    answer: str
    source_context: str
    source_page_numbers: list[int]
    # Populated after RAGAS evaluation
    evaluation_scores: dict[str, float]   # e.g. {"faithfulness": 0.9, ...}
    overall_accuracy: float               # aggregated 0-1 score
    is_faulty: bool                       # True if overall_accuracy < 0.8


# ---------------------------------------------------------------------------
# Page representation carried inside the state
# ---------------------------------------------------------------------------

class PageData(TypedDict):
    page_no: int
    text: str


# ---------------------------------------------------------------------------
# Main LangGraph shared state
# ---------------------------------------------------------------------------

class GraphState(TypedDict):
    # ── Identity / Auth ──────────────────────────────────────────────
    project_id: str           # UUID of the owning Project row
    document_id: str          # UUID of the Document row being processed
    doc_id: str               # Human-readable label, e.g. "DOC-A"

    # ── Full page corpus for this document ───────────────────────────
    all_pages: list[PageData]  # Every page loaded from DB for this doc

    # ── Iteration pointers ───────────────────────────────────────────
    batch_index: int           # Monotonically increasing batch counter
    current_page_start: int    # Index into all_pages where next batch begins
    batch_size: int            # How many pages per batch (default 5)
    overlap: int               # How many pages to share with prev batch (default 1)

    # ── Current batch working data ───────────────────────────────────
    current_batch_pages: list[PageData]   # Pages selected for this batch
    current_batch_id: Optional[str]       # UUID of the saved Batch row
    batch_context: str                    # LLM-condensed context for this batch
    previous_batch_context: str          # Context from the prior batch (multi-hop)

    # ── QA generation / evaluation ───────────────────────────────────
    generated_dataset: list[QAPair]       # QA pairs produced but not yet validated
    validated_dataset: list[QAPair]       # QA pairs that passed RAGAS threshold
    regeneration_attempts: int            # Safety counter – don't loop forever

    # ── Terminal condition flag ───────────────────────────────────────
    is_done: bool                         # True when all pages have been batched


def initial_state(
    project_id: str,
    document_id: str,
    doc_id: str,
    all_pages: list[PageData],
    batch_size: int = 5,
    overlap: int = 1,
) -> GraphState:
    """Bootstrap a fresh GraphState for a new document processing run."""
    return GraphState(
        project_id=project_id,
        document_id=document_id,
        doc_id=doc_id,
        all_pages=all_pages,
        batch_index=0,
        current_page_start=0,
        batch_size=batch_size,
        overlap=overlap,
        current_batch_pages=[],
        current_batch_id=None,
        batch_context="",
        previous_batch_context="",
        generated_dataset=[],
        validated_dataset=[],
        regeneration_attempts=0,
        is_done=False,
    )

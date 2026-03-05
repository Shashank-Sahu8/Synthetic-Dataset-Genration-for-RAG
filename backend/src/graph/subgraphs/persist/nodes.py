"""
Dataset Persist Subgraph – nodes.

Node 1: persist_dataset_node
    Write all evaluated QA pairs (valid + flagged faulty) as DatasetEntry rows.
    Advance the page pointer so the parent graph knows where the next batch starts.

Routing: should_continue_pipeline
    If more pages remain → "batch_context"  (loop in parent graph)
    Otherwise            → "end"
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from sqlalchemy.orm import Session

from backend.database import DatasetEntry
from backend.database.session import SessionLocal
from backend.src.graph.state import GraphState, QAPair

logger = logging.getLogger(__name__)


def persist_dataset_node(state: GraphState) -> dict[str, Any]:
    """
    Persist all QA pairs (valid and faulty) as DatasetEntry rows.
    Advance batch_index and current_page_start for the next iteration.
    """
    generated:   list[QAPair] = state.get("generated_dataset", [])
    valid_pairs  = [p for p in generated if not p["is_faulty"]]
    faulty_pairs = [p for p in generated if p["is_faulty"]]

    db: Session = SessionLocal()
    try:
        for pair in valid_pairs + faulty_pairs:
            db.add(DatasetEntry(
                id=uuid.uuid4(),
                project_id=uuid.UUID(state["project_id"]),
                document_id=uuid.UUID(state["document_id"]),
                batch_id=uuid.UUID(state["current_batch_id"]),
                question=pair["question"],
                answer=pair["answer"],
                source_context=pair["source_context"],
                source_page_numbers=pair["source_page_numbers"],
                evaluation_scores=pair["evaluation_scores"],
                overall_accuracy=pair["overall_accuracy"],
                is_faulty=pair["is_faulty"],
            ))
        db.commit()
    finally:
        db.close()

    logger.info(
        "Persisted  batch_index=%d  doc_id=%s  valid=%d  faulty=%d",
        state["batch_index"], state["doc_id"], len(valid_pairs), len(faulty_pairs),
    )

    overlap:    int = state["overlap"]
    batch_size: int = state["batch_size"]
    next_start: int = state["current_page_start"] + batch_size - overlap

    return {
        "validated_dataset":     state.get("validated_dataset", []) + valid_pairs,
        "generated_dataset":     [],
        "regeneration_attempts": 0,
        "batch_index":           state["batch_index"] + 1,
        "current_page_start":    next_start,
    }


def should_continue_pipeline(state: GraphState) -> str:
    """
    Routing helper used by the parent graph after persist_dataset_node.
    Returns "batch_context" to process the next batch, or "end" when done.
    """
    if state.get("is_done", False):
        return "end"
    if state["current_page_start"] >= len(state["all_pages"]):
        return "end"
    return "batch_context"

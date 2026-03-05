"""
Full LangGraph pipeline: Batch Formation → QA Generation → RAGAS Evaluation → Persist.

Graph topology
──────────────
START
  ↓
create_batch              (slice pages, persist Batch row)
  ↓
create_batch_context      (LLM dense context, persist to DB)
  ↓
generate_dataset          (LLM produces N QA pairs)
  ↓
evaluate_dataset          (RAGAS faithfulness scoring)
  ↓ (conditional: should_regenerate_or_persist)
  ├─ "regenerate"  ──────────────────────→ generate_dataset  (max 3 attempts)
  └─ "persist"     → persist_dataset
                     ↓ (conditional: should_continue_pipeline)
                     ├─ "create_batch" → create_batch  (next batch, loop)
                     └─ "end"          → END
"""
from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import StateGraph, END

from backend.src.graph.state import GraphState, PageData, initial_state
from backend.src.graph.nodes import (
    create_batch_node,
    create_batch_context_node,
    generate_dataset_node,
    evaluate_dataset_node,
    persist_dataset_node,
    should_regenerate_or_persist,
    should_continue_pipeline,
)

logger = logging.getLogger(__name__)


def _build_graph() -> Any:
    """Compile and return the full LangGraph StateGraph."""
    builder = StateGraph(GraphState)

    # ── Register all nodes ────────────────────────────────────────────────
    builder.add_node("create_batch",         create_batch_node)
    builder.add_node("create_batch_context", create_batch_context_node)
    builder.add_node("generate_dataset",     generate_dataset_node)
    builder.add_node("evaluate_dataset",     evaluate_dataset_node)
    builder.add_node("persist_dataset",      persist_dataset_node)

    # ── Entry point ───────────────────────────────────────────────────────
    builder.set_entry_point("create_batch")

    # ── Phase 1: fixed edges ──────────────────────────────────────────────
    builder.add_edge("create_batch",         "create_batch_context")
    builder.add_edge("create_batch_context", "generate_dataset")

    # ── Phase 2 → 3: fixed edge ───────────────────────────────────────────
    builder.add_edge("generate_dataset", "evaluate_dataset")

    # ── Phase 3 → conditional: regenerate or persist ──────────────────────
    builder.add_conditional_edges(
        "evaluate_dataset",
        should_regenerate_or_persist,
        {
            "regenerate": "generate_dataset",
            "persist":    "persist_dataset",
        },
    )

    # ── Phase 4 → conditional: next batch or end ──────────────────────────
    builder.add_conditional_edges(
        "persist_dataset",
        should_continue_pipeline,
        {
            "create_batch": "create_batch",
            "end":          END,
        },
    )

    return builder.compile()


# Compile once at import time
_graph = _build_graph()


def run_full_pipeline(
    project_id: str,
    document_id: str,
    doc_id: str,
    pages: list[dict],          # list of {"page_no": int, "text": str}
    batch_size: int = 5,
    overlap: int = 1,
) -> GraphState:
    """
    Public entry-point called by the ingest route's background thread.

    Runs the complete pipeline (Phase 1-4) synchronously within the background
    thread until all pages have been processed and DatasetEntry rows persisted.

    Returns the final GraphState after the graph terminates.
    """
    page_data: list[PageData] = [
        {"page_no": p["page_no"], "text": p["text"]} for p in pages
    ]

    state: GraphState = initial_state(
        project_id=project_id,
        document_id=document_id,
        doc_id=doc_id,
        all_pages=page_data,
        batch_size=batch_size,
        overlap=overlap,
    )

    logger.info(
        "Full pipeline START  doc_id=%s  total_pages=%d  batch_size=%d  overlap=%d",
        doc_id,
        len(page_data),
        batch_size,
        overlap,
    )

    final_state: GraphState = _graph.invoke(state)

    logger.info(
        "Full pipeline END  doc_id=%s  validated_qa_pairs=%d",
        doc_id,
        len(final_state.get("validated_dataset", [])),
    )

    return final_state


# Keep backward-compat alias so any code still importing run_phase1_pipeline works
run_phase1_pipeline = run_full_pipeline

"""
Parent LangGraph pipeline: orchestrates the three subgraphs.

Each subgraph handles one phase of the pipeline:
  ┌─────────────────────────────────────────────────────────────────────┐
  │  batch_context_subgraph                                             │
  │  START → create_batch → create_batch_context → END                 │
  └─────────────────────────────────────────────────────────────────────┘
            ↓
  ┌─────────────────────────────────────────────────────────────────────┐
  │  qa_generation_subgraph                                             │
  │  START → generate_dataset → evaluate_dataset → (regenerate|END)    │
  └─────────────────────────────────────────────────────────────────────┘
            ↓
  ┌─────────────────────────────────────────────────────────────────────┐
  │  persist_subgraph                                                   │
  │  START → persist_dataset → END                                      │
  └─────────────────────────────────────────────────────────────────────┘
            ↓ (conditional: should_continue_pipeline)
            ├─ "batch_context" ──→ batch_context_subgraph  (next batch)
            └─ "end"           ──→ END
"""
from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import StateGraph, END

from backend.src.graph.state import GraphState, PageData, initial_state
from backend.src.graph.subgraphs.batch_context import build_batch_context_subgraph
from backend.src.graph.subgraphs.qa_generation import build_qa_generation_subgraph
from backend.src.graph.subgraphs.persist import build_persist_subgraph
from backend.src.graph.subgraphs.persist.nodes import should_continue_pipeline

logger = logging.getLogger(__name__)


def _build_parent_graph() -> Any:
    """
    Compile the parent graph that wires the three subgraphs together.

    The subgraphs are compiled once and added as opaque nodes; all three
    share the same GraphState TypedDict so state flows through seamlessly.
    """
    batch_context_sg = build_batch_context_subgraph()
    qa_generation_sg = build_qa_generation_subgraph()
    persist_sg       = build_persist_subgraph()

    builder = StateGraph(GraphState)

    # ── Register compiled subgraphs as nodes ─────────────────────────────────
    builder.add_node("batch_context",  batch_context_sg)
    builder.add_node("qa_generation",  qa_generation_sg)
    builder.add_node("persist",        persist_sg)

    # ── Entry point ───────────────────────────────────────────────────────────
    builder.set_entry_point("batch_context")

    # ── Fixed edges: batch → qa → persist ────────────────────────────────────
    builder.add_edge("batch_context", "qa_generation")
    builder.add_edge("qa_generation", "persist")

    # ── Conditional edge after persist: next batch or END ────────────────────
    builder.add_conditional_edges(
        "persist",
        should_continue_pipeline,
        {
            "batch_context": "batch_context",
            "end":           END,
        },
    )

    return builder.compile()


# Compile once at import time
_graph = _build_parent_graph()


def run_full_pipeline(
    project_id:  str,
    document_id: str,
    doc_id:      str,
    pages:       list[dict],   # [{"page_no": int, "text": str}]
    batch_size:  int = 5,
    overlap:     int = 1,
) -> GraphState:
    """
    Public entry-point called by the ingest route's background thread.

    Executes the complete pipeline (Phases 1-4) synchronously, iterating
    over every batch until all pages are processed and DatasetEntry rows
    are persisted.

    Returns the final GraphState.
    """
    page_data: list[PageData] = [
        {"page_no": p["page_no"], "text": p["text"]} for p in pages
    ]

    state = initial_state(
        project_id=project_id,
        document_id=document_id,
        doc_id=doc_id,
        all_pages=page_data,
        batch_size=batch_size,
        overlap=overlap,
    )

    logger.info(
        "Pipeline START  doc_id=%s  total_pages=%d  batch_size=%d  overlap=%d",
        doc_id, len(page_data), batch_size, overlap,
    )

    final_state: GraphState = _graph.invoke(state)

    logger.info(
        "Pipeline END  doc_id=%s  validated_qa_pairs=%d",
        doc_id, len(final_state.get("validated_dataset", [])),
    )

    return final_state


# Backward-compat alias
run_phase1_pipeline = run_full_pipeline

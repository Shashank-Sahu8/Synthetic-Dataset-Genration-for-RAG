"""
Batch Context Subgraph – compiled LangGraph subgraph.

Topology
────────
START
  ↓
create_batch          (slice pages, persist empty Batch row)
  ↓
create_batch_context  (LLM dense knowledge summary, persist to DB)
  ↓
END
"""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from backend.src.graph.state import GraphState
from backend.src.graph.subgraphs.batch_context.nodes import (
    create_batch_node,
    create_batch_context_node,
)


def build_batch_context_subgraph():
    """Build and compile the Batch Context subgraph."""
    builder = StateGraph(GraphState)

    builder.add_node("create_batch",         create_batch_node)
    builder.add_node("create_batch_context", create_batch_context_node)

    builder.set_entry_point("create_batch")
    builder.add_edge("create_batch",         "create_batch_context")
    builder.add_edge("create_batch_context", END)

    return builder.compile()

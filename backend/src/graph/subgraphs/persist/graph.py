"""
Dataset Persist Subgraph – compiled LangGraph subgraph.

Topology
────────
START
  ↓
persist_dataset  (write DatasetEntry rows, advance page pointer)
  ↓
END
"""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from backend.src.graph.state import GraphState
from backend.src.graph.subgraphs.persist.nodes import persist_dataset_node


def build_persist_subgraph():
    """Build and compile the Dataset Persist subgraph."""
    builder = StateGraph(GraphState)

    builder.add_node("persist_dataset", persist_dataset_node)

    builder.set_entry_point("persist_dataset")
    builder.add_edge("persist_dataset", END)

    return builder.compile()

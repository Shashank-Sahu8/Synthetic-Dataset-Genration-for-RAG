"""
QA Generation + Evaluation Subgraph – compiled LangGraph subgraph.

Topology
────────
START
  ↓
generate_dataset    (LLM produces N QA pairs from batch context)
  ↓
evaluate_dataset    (RAGAS faithfulness scoring)
  ↓ (conditional: should_regenerate_or_persist)
  ├─ "regenerate" ──→ generate_dataset   (up to 3 retries)
  └─ "persist"    ──→ END
"""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from backend.src.graph.state import GraphState
from backend.src.graph.subgraphs.qa_generation.nodes import (
    generate_dataset_node,
    evaluate_dataset_node,
    should_regenerate_or_persist,
)


def build_qa_generation_subgraph():
    """Build and compile the QA Generation + Evaluation subgraph."""
    builder = StateGraph(GraphState)

    builder.add_node("generate_dataset", generate_dataset_node)
    builder.add_node("evaluate_dataset", evaluate_dataset_node)

    builder.set_entry_point("generate_dataset")
    builder.add_edge("generate_dataset", "evaluate_dataset")

    builder.add_conditional_edges(
        "evaluate_dataset",
        should_regenerate_or_persist,
        {
            "regenerate": "generate_dataset",
            "persist":    END,
        },
    )

    return builder.compile()

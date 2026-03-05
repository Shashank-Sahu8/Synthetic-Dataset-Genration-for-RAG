"""
Batch Context Subgraph – nodes.

Node 1: create_batch_node
    Slice the next overlapping window of pages from all_pages.
    Persist an empty Batch row to PostgreSQL.

Node 2: create_batch_context_node
    Concatenate raw page text and call the LLM for a dense knowledge summary.
    Update the Batch.batch_context column.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from sqlalchemy.orm import Session

from backend.database import Batch
from backend.database.session import SessionLocal
from backend.src.graph.llm import llm_call
from backend.src.graph.state import GraphState, PageData
from backend.src.prompts import BATCH_CONTEXT_SYSTEM, BATCH_CONTEXT_USER

logger = logging.getLogger(__name__)


def create_batch_node(state: GraphState) -> dict[str, Any]:
    """
    Slice the next overlapping batch of pages from all_pages.
    Persist an empty Batch row and return updated state keys.
    """
    all_pages: list[PageData] = state["all_pages"]
    start: int     = state["current_page_start"]
    batch_size: int = state["batch_size"]

    if start >= len(all_pages):
        return {"is_done": True}

    batch_pages = all_pages[start : start + batch_size]

    db: Session = SessionLocal()
    try:
        batch_row = Batch(
            id=uuid.uuid4(),
            project_id=uuid.UUID(state["project_id"]),
            document_id=uuid.UUID(state["document_id"]),
            batch_index=state["batch_index"],
            page_ids=[],
            batch_context=None,
        )
        db.add(batch_row)
        db.commit()
        db.refresh(batch_row)
        batch_id = str(batch_row.id)
    finally:
        db.close()

    logger.info(
        "Batch %d created  doc_id=%s  pages=%d-%d  batch_db_id=%s",
        state["batch_index"],
        state["doc_id"],
        all_pages[start]["page_no"],
        batch_pages[-1]["page_no"],
        batch_id,
    )

    return {
        "current_batch_pages": batch_pages,
        "current_batch_id":    batch_id,
        "generated_dataset":   [],
        "regeneration_attempts": 0,
        "is_done":             False,
    }


def create_batch_context_node(state: GraphState) -> dict[str, Any]:
    """
    Build a dense knowledge summary of the batch via LLM and persist it.
    """
    batch_pages: list[PageData] = state["current_batch_pages"]

    raw_text   = "\n\n".join(f"[Page {p['page_no']}]\n{p['text']}" for p in batch_pages)
    page_range = f"{batch_pages[0]['page_no']}-{batch_pages[-1]['page_no']}"

    batch_context = llm_call([
        {"role": "system", "content": BATCH_CONTEXT_SYSTEM},
        {"role": "user",   "content": BATCH_CONTEXT_USER.format(
            page_range=page_range,
            raw_text=raw_text,
        )},
    ])

    db: Session = SessionLocal()
    try:
        batch_row = (
            db.query(Batch)
            .filter(Batch.id == uuid.UUID(state["current_batch_id"]))
            .first()
        )
        if batch_row:
            batch_row.batch_context = batch_context
            db.commit()
    finally:
        db.close()

    logger.info(
        "Batch context saved  batch_index=%d  doc_id=%s  context_len=%d",
        state["batch_index"],
        state["doc_id"],
        len(batch_context),
    )

    return {
        "batch_context":          batch_context,
        "previous_batch_context": batch_context,
    }

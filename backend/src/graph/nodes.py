"""
Full LangGraph pipeline nodes (Phase 1 → 4).

Phase 1 – Batch formation
  Node 1: create_batch_node        – slice pages, persist Batch row
  Node 2: create_batch_context_node– LLM dense knowledge summary, persist

Phase 2 – QA Dataset Generation
  Node 3: generate_dataset_node    – LLM produces N Q&A pairs per batch

Phase 3 – RAGAS Evaluation
  Node 4: evaluate_dataset_node    – score pairs with RAGAS faithfulness;
                                     mark is_faulty based on threshold

Phase 4 – Persist
  Node 5: persist_dataset_node     – write validated DatasetEntry rows to DB

Routing helpers
  should_regenerate_or_persist     – after evaluate: retry or persist
  should_continue_pipeline         – after persist: next batch or END

All LLM calls use litellm via OpenRouter.
"""
from __future__ import annotations

import json
import logging
import os
import re
import uuid
from typing import Any

import litellm
from sqlalchemy.orm import Session

from backend.database import Batch, DatasetEntry
from backend.database.session import SessionLocal
from backend.src.graph.state import GraphState, PageData, QAPair
from backend.src.prompts import (
    BATCH_CONTEXT_SYSTEM,
    BATCH_CONTEXT_USER,
    QA_GENERATOR_SYSTEM,
    QA_GENERATOR_USER,
    QA_REGENERATOR_SYSTEM,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# litellm helper
# ─────────────────────────────────────────────────────────────────────────────

_LLM_MODEL: str = os.getenv("LLM_MODEL", "openrouter/meta-llama/llama-3.3-70b-instruct")
_LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
_LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
_LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))

# Silence verbose litellm logs
litellm.set_verbose = False


def _llm_call(
    messages: list[dict[str, str]],
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """
    Single wrapper for every litellm call in the pipeline.

    Args:
        messages: OpenAI-style message list [{"role": ..., "content": ...}]
        temperature: override default if provided
        max_tokens: override default if provided

    Returns:
        The model's text response (stripped).
    """
    response = litellm.completion(
        model=_LLM_MODEL,
        messages=messages,
        api_key=_LLM_API_KEY,
        temperature=temperature if temperature is not None else _LLM_TEMPERATURE,
        max_tokens=max_tokens if max_tokens is not None else _LLM_MAX_TOKENS,
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_json_array(raw: str) -> list[dict]:
    """
    Robustly extract a JSON array from an LLM response that may wrap it in
    markdown fences or add trailing commentary.
    """
    # Strip markdown fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

    # Find the outermost [ ... ] block
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON array found in LLM output:\n{raw[:500]}")

    return json.loads(cleaned[start : end + 1])


def _num_questions_for_batch(batch_pages: list[PageData]) -> int:
    """Return a sensible number of QA pairs for the given batch size."""
    return max(3, len(batch_pages) * 2)


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 – create_batch_node (Phase 1)
# ─────────────────────────────────────────────────────────────────────────────

def create_batch_node(state: GraphState) -> dict[str, Any]:
    """
    Slice the next overlapping batch of pages from all_pages.
    Persist an empty Batch row to the DB and return updated state keys.
    """
    all_pages: list[PageData] = state["all_pages"]
    start: int = state["current_page_start"]
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
        "current_batch_id": batch_id,
        "generated_dataset": [],
        "regeneration_attempts": 0,
        "is_done": False,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 – create_batch_context_node (Phase 1)
# ─────────────────────────────────────────────────────────────────────────────

def create_batch_context_node(state: GraphState) -> dict[str, Any]:
    """
    Concatenate raw page text, call the LLM for a dense knowledge summary,
    and persist it back to the Batch row.
    """
    batch_pages: list[PageData] = state["current_batch_pages"]

    raw_text = "\n\n".join(
        f"[Page {p['page_no']}]\n{p['text']}" for p in batch_pages
    )
    page_range = f"{batch_pages[0]['page_no']}-{batch_pages[-1]['page_no']}"

    batch_context = _llm_call([
        {"role": "system", "content": BATCH_CONTEXT_SYSTEM},
        {"role": "user",   "content": BATCH_CONTEXT_USER.format(
            page_range=page_range,
            raw_text=raw_text,
        )},
    ])

    # Persist context to DB
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
        "batch_context": batch_context,
        "previous_batch_context": batch_context,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 – generate_dataset_node (Phase 2)
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset_node(state: GraphState) -> dict[str, Any]:
    """
    Generate QA pairs from the current batch context using the LLM.

    On the first attempt uses the standard generator prompt.
    On subsequent attempts (regeneration) uses the stricter regenerator prompt.
    """
    batch_pages: list[PageData] = state["current_batch_pages"]
    batch_context: str = state["batch_context"]
    previous_batch_context: str = state.get("previous_batch_context", "")
    regeneration_attempts: int = state.get("regeneration_attempts", 0)

    page_range = f"{batch_pages[0]['page_no']}-{batch_pages[-1]['page_no']}"
    num_questions = _num_questions_for_batch(batch_pages)

    # Choose system prompt based on attempt number
    system_prompt = (
        QA_REGENERATOR_SYSTEM.format(num_questions=num_questions)
        if regeneration_attempts > 0
        else QA_GENERATOR_SYSTEM.format(num_questions=num_questions)
    )

    user_prompt = QA_GENERATOR_USER.format(
        page_range=page_range,
        batch_context=batch_context,
        previous_batch_context=previous_batch_context or "(none)",
        num_questions=num_questions,
    )

    raw_output = _llm_call(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.4 if regeneration_attempts > 0 else _LLM_TEMPERATURE,
    )

    try:
        raw_pairs = _parse_json_array(raw_output)
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning(
            "JSON parse error in generate_dataset_node  attempt=%d  error=%s",
            regeneration_attempts,
            exc,
        )
        raw_pairs = []

    qa_pairs: list[QAPair] = []
    for item in raw_pairs:
        if not isinstance(item, dict):
            continue
        qa_pairs.append(
            QAPair(
                question=str(item.get("question", "")).strip(),
                answer=str(item.get("answer", "")).strip(),
                source_context=str(item.get("source_context", batch_context[:300])).strip(),
                source_page_numbers=item.get("source_page_numbers", [p["page_no"] for p in batch_pages]),
                evaluation_scores={},
                overall_accuracy=0.0,
                is_faulty=True,  # default until evaluated
            )
        )

    logger.info(
        "QA generation  batch_index=%d  doc_id=%s  attempt=%d  pairs=%d",
        state["batch_index"],
        state["doc_id"],
        regeneration_attempts,
        len(qa_pairs),
    )

    return {
        "generated_dataset": qa_pairs,
        "regeneration_attempts": regeneration_attempts + 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 4 – evaluate_dataset_node (Phase 3)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_dataset_node(state: GraphState) -> dict[str, Any]:
    """
    Evaluate generated QA pairs using RAGAS Faithfulness metric.

    Each pair is scored against its source_context. Pairs scoring below
    FAITHFULNESS_THRESHOLD are marked is_faulty=True.
    """
    FAITHFULNESS_THRESHOLD = 0.7

    generated: list[QAPair] = state.get("generated_dataset", [])
    if not generated:
        logger.warning("evaluate_dataset_node called with empty generated_dataset")
        return {"generated_dataset": []}

    try:
        scored_pairs = _ragas_evaluate(generated, FAITHFULNESS_THRESHOLD)
    except Exception as exc:
        logger.warning(
            "RAGAS evaluation failed (%s) – falling back to LLM-as-judge", exc
        )
        scored_pairs = _llm_judge_evaluate(generated, FAITHFULNESS_THRESHOLD)

    faulty_count = sum(1 for p in scored_pairs if p["is_faulty"])
    logger.info(
        "Evaluation  batch_index=%d  doc_id=%s  total=%d  faulty=%d",
        state["batch_index"],
        state["doc_id"],
        len(scored_pairs),
        faulty_count,
    )

    return {"generated_dataset": scored_pairs}


def _ragas_evaluate(
    pairs: list[QAPair],
    threshold: float,
) -> list[QAPair]:
    """Run RAGAS Faithfulness evaluation on a list of QA pairs."""
    from datasets import Dataset as HFDataset
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import faithfulness
    from ragas.llms import LangchainLLMWrapper
    from langchain_community.chat_models import ChatLiteLLM

    ragas_llm = LangchainLLMWrapper(
        ChatLiteLLM(
            model=_LLM_MODEL,
            api_key=_LLM_API_KEY,
        )
    )
    faithfulness.llm = ragas_llm  # type: ignore[attr-defined]

    hf_data = HFDataset.from_list([
        {
            "question":  p["question"],
            "answer":    p["answer"],
            "contexts":  [p["source_context"]],
        }
        for p in pairs
    ])

    result = ragas_evaluate(hf_data, metrics=[faithfulness])
    result_df = result.to_pandas()

    scored: list[QAPair] = []
    for i, pair in enumerate(pairs):
        faith_score: float = float(result_df.iloc[i].get("faithfulness", 0.0))
        overall = round(faith_score, 4)
        updated = QAPair(
            **{
                **pair,
                "evaluation_scores": {"faithfulness": faith_score},
                "overall_accuracy": overall,
                "is_faulty": overall < threshold,
            }
        )
        scored.append(updated)

    return scored


def _llm_judge_evaluate(
    pairs: list[QAPair],
    threshold: float,
) -> list[QAPair]:
    """
    Fallback: use the LLM itself to rate each answer's faithfulness (0-1).
    Cheaper than RAGAS but still meaningful for filtering.
    """
    JUDGE_SYSTEM = (
        "You are a strict factual accuracy judge. "
        "Given a context, a question, and an answer, output ONLY a JSON object "
        '{"score": <float 0.0-1.0>} representing how faithful the answer is to '
        "the context. 1.0 = fully grounded, 0.0 = hallucinated."
    )

    scored: list[QAPair] = []
    for pair in pairs:
        prompt = (
            f"CONTEXT:\n{pair['source_context']}\n\n"
            f"QUESTION: {pair['question']}\n"
            f"ANSWER: {pair['answer']}\n\n"
            "Rate faithfulness:"
        )
        try:
            raw = _llm_call(
                [
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.0,
                max_tokens=64,
            )
            cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
            score = float(json.loads(cleaned).get("score", 0.0))
        except Exception:
            score = 0.5  # neutral default on parse error

        overall = round(max(0.0, min(1.0, score)), 4)
        scored.append(
            QAPair(
                **{
                    **pair,
                    "evaluation_scores": {"faithfulness": overall},
                    "overall_accuracy": overall,
                    "is_faulty": overall < threshold,
                }
            )
        )

    return scored


# ─────────────────────────────────────────────────────────────────────────────
# Node 5 – persist_dataset_node (Phase 4)
# ─────────────────────────────────────────────────────────────────────────────

def persist_dataset_node(state: GraphState) -> dict[str, Any]:
    """
    Persist all non-faulty QA pairs as DatasetEntry rows.
    Advances page pointers so the next batch can be started.
    """
    generated: list[QAPair] = state.get("generated_dataset", [])
    valid_pairs = [p for p in generated if not p["is_faulty"]]
    faulty_pairs = [p for p in generated if p["is_faulty"]]

    db: Session = SessionLocal()
    try:
        for pair in valid_pairs:
            entry = DatasetEntry(
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
                is_faulty=False,
            )
            db.add(entry)

        # Persist faulty entries too (for analysis), flagged is_faulty=True
        for pair in faulty_pairs:
            entry = DatasetEntry(
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
                is_faulty=True,
            )
            db.add(entry)

        db.commit()
    finally:
        db.close()

    logger.info(
        "Persisted  batch_index=%d  doc_id=%s  valid=%d  faulty=%d",
        state["batch_index"],
        state["doc_id"],
        len(valid_pairs),
        len(faulty_pairs),
    )

    # Advance page pointer
    overlap: int = state["overlap"]
    batch_size: int = state["batch_size"]
    next_start: int = state["current_page_start"] + batch_size - overlap

    return {
        "validated_dataset": state.get("validated_dataset", []) + valid_pairs,
        "generated_dataset": [],
        "regeneration_attempts": 0,
        "batch_index": state["batch_index"] + 1,
        "current_page_start": next_start,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Routing helpers
# ─────────────────────────────────────────────────────────────────────────────

MAX_REGENERATION_ATTEMPTS = 3
FAULTY_RATE_THRESHOLD = 0.60  # if >60 % of pairs are faulty → regenerate


def should_regenerate_or_persist(state: GraphState) -> str:
    """
    After evaluation decide: retry generation or persist what we have.

    Returns "regenerate" or "persist".
    """
    generated: list[QAPair] = state.get("generated_dataset", [])
    attempts: int = state.get("regeneration_attempts", 0)

    if not generated:
        return "persist"

    faulty_rate = sum(1 for p in generated if p["is_faulty"]) / len(generated)

    if faulty_rate > FAULTY_RATE_THRESHOLD and attempts < MAX_REGENERATION_ATTEMPTS:
        logger.info(
            "Regenerating  batch_index=%d  faulty_rate=%.0f%%  attempt=%d",
            state["batch_index"],
            faulty_rate * 100,
            attempts,
        )
        return "regenerate"

    return "persist"


def should_continue_pipeline(state: GraphState) -> str:
    """
    After persisting a batch: start the next batch or end the pipeline.

    Returns "create_batch" or "end".
    """
    if state.get("is_done", False):
        return "end"
    if state["current_page_start"] >= len(state["all_pages"]):
        return "end"
    return "create_batch"

"""
QA Generation + Evaluation Subgraph – nodes.

Node 1: generate_dataset_node
    Call the LLM to produce N QA pairs from the current batch context.
    Uses the standard generator prompt on the first attempt and the stricter
    regenerator prompt on retry attempts.

Node 2: evaluate_dataset_node
    Score each QA pair using RAGAS Faithfulness (falls back to LLM-as-judge
    when RAGAS is unavailable). Mark pairs is_faulty=True when score < threshold.

Routing: should_regenerate_or_persist
    If >60 % of pairs are faulty AND attempts < 3 → "regenerate"
    Otherwise                                       → "persist"
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from backend.src.graph.llm import llm_call, parse_json_array, LLM_MODEL, LLM_API_KEY, LLM_TEMPERATURE
from backend.src.graph.state import GraphState, PageData, QAPair
from backend.src.prompts import (
    QA_GENERATOR_SYSTEM,
    QA_GENERATOR_USER,
    QA_REGENERATOR_SYSTEM,
)

logger = logging.getLogger(__name__)

MAX_REGENERATION_ATTEMPTS = 3
FAULTY_RATE_THRESHOLD     = 0.60   # >60 % faulty → regenerate
FAITHFULNESS_THRESHOLD    = 0.70   # per-pair pass/fail threshold


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _num_questions(batch_pages: list[PageData]) -> int:
    return max(3, len(batch_pages) * 2)


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 – generate_dataset_node
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset_node(state: GraphState) -> dict[str, Any]:
    """
    Generate QA pairs from the current batch context.
    Uses stricter prompt on regeneration attempts.
    """
    batch_pages:           list[PageData] = state["current_batch_pages"]
    batch_context:         str            = state["batch_context"]
    previous_batch_context: str           = state.get("previous_batch_context", "")
    regeneration_attempts: int            = state.get("regeneration_attempts", 0)

    page_range    = f"{batch_pages[0]['page_no']}-{batch_pages[-1]['page_no']}"
    num_questions = _num_questions(batch_pages)

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

    raw_output = llm_call(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.4 if regeneration_attempts > 0 else LLM_TEMPERATURE,
    )

    try:
        raw_pairs = parse_json_array(raw_output)
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning(
            "JSON parse error in generate_dataset_node  attempt=%d  error=%s",
            regeneration_attempts, exc,
        )
        raw_pairs = []

    qa_pairs: list[QAPair] = []
    for item in raw_pairs:
        if not isinstance(item, dict):
            continue
        qa_pairs.append(QAPair(
            question=str(item.get("question", "")).strip(),
            answer=str(item.get("answer", "")).strip(),
            source_context=str(item.get("source_context", batch_context[:300])).strip(),
            source_page_numbers=item.get("source_page_numbers",
                                         [p["page_no"] for p in batch_pages]),
            evaluation_scores={},
            overall_accuracy=0.0,
            is_faulty=True,
        ))

    logger.info(
        "QA generation  batch_index=%d  doc_id=%s  attempt=%d  pairs=%d",
        state["batch_index"], state["doc_id"], regeneration_attempts, len(qa_pairs),
    )

    return {
        "generated_dataset":     qa_pairs,
        "regeneration_attempts": regeneration_attempts + 1,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 – evaluate_dataset_node
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_dataset_node(state: GraphState) -> dict[str, Any]:
    """
    Score QA pairs with RAGAS Faithfulness; fallback to LLM-as-judge.
    """
    generated: list[QAPair] = state.get("generated_dataset", [])
    if not generated:
        logger.warning("evaluate_dataset_node called with empty generated_dataset")
        return {"generated_dataset": []}

    try:
        scored = _ragas_evaluate(generated)
    except Exception as exc:
        logger.warning("RAGAS evaluation failed (%s) – falling back to LLM-as-judge", exc)
        scored = _llm_judge_evaluate(generated)

    faulty_count = sum(1 for p in scored if p["is_faulty"])
    logger.info(
        "Evaluation  batch_index=%d  doc_id=%s  total=%d  faulty=%d",
        state["batch_index"], state["doc_id"], len(scored), faulty_count,
    )
    return {"generated_dataset": scored}


def _ragas_evaluate(pairs: list[QAPair]) -> list[QAPair]:
    """RAGAS Faithfulness scorer."""
    from datasets import Dataset as HFDataset
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import faithfulness
    from ragas.llms import LangchainLLMWrapper
    from langchain_community.chat_models import ChatLiteLLM

    faithfulness.llm = LangchainLLMWrapper(
        ChatLiteLLM(model=LLM_MODEL, api_key=LLM_API_KEY)
    )

    hf_data = HFDataset.from_list([
        {"question": p["question"], "answer": p["answer"], "contexts": [p["source_context"]]}
        for p in pairs
    ])

    result_df = ragas_evaluate(hf_data, metrics=[faithfulness]).to_pandas()

    scored: list[QAPair] = []
    for i, pair in enumerate(pairs):
        faith_score = float(result_df.iloc[i].get("faithfulness", 0.0))
        overall     = round(faith_score, 4)
        scored.append(QAPair(**{
            **pair,
            "evaluation_scores": {"faithfulness": faith_score},
            "overall_accuracy":  overall,
            "is_faulty":         overall < FAITHFULNESS_THRESHOLD,
        }))
    return scored


def _llm_judge_evaluate(pairs: list[QAPair]) -> list[QAPair]:
    """LLM-as-judge fallback scorer."""
    JUDGE_SYSTEM = (
        "You are a strict factual accuracy judge. "
        "Given a context, a question, and an answer, output ONLY a JSON object "
        '{"score": <float 0.0-1.0>} where 1.0=fully grounded, 0.0=hallucinated.'
    )
    scored: list[QAPair] = []
    for pair in pairs:
        prompt = (
            f"CONTEXT:\n{pair['source_context']}\n\n"
            f"QUESTION: {pair['question']}\n"
            f"ANSWER: {pair['answer']}\n\nRate faithfulness:"
        )
        try:
            raw     = llm_call(
                [{"role": "system", "content": JUDGE_SYSTEM},
                 {"role": "user",   "content": prompt}],
                temperature=0.0, max_tokens=64,
            )
            cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
            score   = float(json.loads(cleaned).get("score", 0.0))
        except Exception:
            score = 0.5

        overall = round(max(0.0, min(1.0, score)), 4)
        scored.append(QAPair(**{
            **pair,
            "evaluation_scores": {"faithfulness": overall},
            "overall_accuracy":  overall,
            "is_faulty":         overall < FAITHFULNESS_THRESHOLD,
        }))
    return scored


# ─────────────────────────────────────────────────────────────────────────────
# Routing helper
# ─────────────────────────────────────────────────────────────────────────────

def should_regenerate_or_persist(state: GraphState) -> str:
    """
    After evaluation decide: retry generation or move to persist.
    Returns "regenerate" or "persist".
    """
    generated: list[QAPair] = state.get("generated_dataset", [])
    attempts:  int           = state.get("regeneration_attempts", 0)

    if not generated:
        return "persist"

    faulty_rate = sum(1 for p in generated if p["is_faulty"]) / len(generated)

    if faulty_rate > FAULTY_RATE_THRESHOLD and attempts < MAX_REGENERATION_ATTEMPTS:
        logger.info(
            "Regenerating  batch_index=%d  faulty_rate=%.0f%%  attempt=%d",
            state["batch_index"], faulty_rate * 100, attempts,
        )
        return "regenerate"

    return "persist"

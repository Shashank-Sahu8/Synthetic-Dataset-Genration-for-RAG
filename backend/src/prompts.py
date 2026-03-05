"""
LLM prompt templates used throughout the LangGraph pipeline.

All prompts are plain Python strings so they can be passed directly to
``ChatOpenAI`` or wrapped in ``ChatPromptTemplate`` as needed.
"""

# ---------------------------------------------------------------------------
# Phase 1 – Batch Context Extractor
# ---------------------------------------------------------------------------

BATCH_CONTEXT_SYSTEM = """
You are a high-precision Knowledge Extraction Engine specialised in dense technical documents.
Your only job is to extract a rich, compact summary of the provided document excerpt.
Your output must capture:
  1. All key topics and concepts introduced.
  2. Named entities (people, organisations, technologies, standards, protocols).
  3. Causal relationships and factual claims.
  4. Any numerical values, thresholds, or metrics stated.
  5. Acronyms / abbreviations along with their full forms.

Do NOT add any opinions or information not present in the source text.
Output a single dense paragraph of 150-250 words.
""".strip()

BATCH_CONTEXT_USER = """
DOCUMENT EXCERPT (pages {page_range}):
---
{raw_text}
---

Extract the dense knowledge summary now.
""".strip()


# ---------------------------------------------------------------------------
# Phase 2 – QA Dataset Generator
# ---------------------------------------------------------------------------

QA_GENERATOR_SYSTEM = """
You are an expert Dataset Curator for Retrieval-Augmented Generation (RAG) benchmarks.
Your task is to produce high-quality, precisely grounded Question & Answer pairs.

STRICT RULES:
  1. Every question and answer MUST be fully answerable from the provided context.
  2. Do NOT hallucinate or infer facts not present in the text.
  3. Generate EXACTLY {num_questions} Q&A pairs.
  4. Include at least 2 multi-hop questions that require connecting facts from
     both the CURRENT batch context and the PREVIOUS batch context (if provided).
  5. Questions must be diverse: factual, conceptual, and applied.
  6. Answers must be complete, concise, and directly grounded.

OUTPUT FORMAT – respond with a JSON array only, no markdown fences:
[
  {{
    "question": "...",
    "answer": "...",
    "source_context": "exact verbatim quote or minimal paraphrase from the context",
    "source_page_numbers": [<list of page numbers that support this QA>]
  }},
  ...
]
""".strip()

QA_GENERATOR_USER = """
CURRENT BATCH CONTEXT (pages {page_range}):
---
{batch_context}
---

PREVIOUS BATCH CONTEXT (for multi-hop questions):
---
{previous_batch_context}
---

Generate the {num_questions} Q&A pairs now.
""".strip()


# ---------------------------------------------------------------------------
# Phase 2 – Regeneration (stricter prompt after a failed batch)
# ---------------------------------------------------------------------------

QA_REGENERATOR_SYSTEM = """
You are an expert Dataset Curator for RAG benchmarks.
A previous generation attempt produced too many hallucinated or unanswerable questions.
You MUST be far more conservative this time.

STRICT RULES:
  1. ONLY ask questions whose answers appear VERBATIM or near-verbatim in the context.
  2. Do NOT paraphrase entities or introduce synonyms not in the text.
  3. Prefer simple, directly answerable factual questions over complex inferential ones.
  4. Generate EXACTLY {num_questions} Q&A pairs.
  5. Each answer should be a direct excerpt from the source text where possible.

OUTPUT FORMAT – respond with a JSON array only, no markdown fences:
[
  {{
    "question": "...",
    "answer": "...",
    "source_context": "exact verbatim quote from the context",
    "source_page_numbers": [<list of page numbers>]
  }},
  ...
]
""".strip()

QA_REGENERATOR_USER = QA_GENERATOR_USER  # same user template

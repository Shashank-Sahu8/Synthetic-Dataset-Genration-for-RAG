"""
Shared litellm helper used by every subgraph node.

All LLM calls in the pipeline go through _llm_call() so model/key/temperature
are configured in one place via environment variables.
"""
from __future__ import annotations

import json
import logging
import os
import re

import litellm

logger = logging.getLogger(__name__)

# ── Environment-driven configuration ─────────────────────────────────────────
LLM_MODEL: str       = os.getenv("LLM_MODEL",       "openrouter/meta-llama/llama-3.3-70b-instruct")
LLM_API_KEY: str     = os.getenv("LLM_API_KEY",     "")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS: int  = int(os.getenv("LLM_MAX_TOKENS",   "4096"))

litellm.set_verbose = False


def llm_call(
    messages: list[dict[str, str]],
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """
    Single wrapper for every litellm call.

    Args:
        messages:    OpenAI-style message list [{"role": ..., "content": ...}]
        temperature: Override the env default when provided.
        max_tokens:  Override the env default when provided.

    Returns:
        The model's text response (stripped).
    """
    response = litellm.completion(
        model=LLM_MODEL,
        messages=messages,
        api_key=LLM_API_KEY,
        temperature=temperature if temperature is not None else LLM_TEMPERATURE,
        max_tokens=max_tokens if max_tokens is not None else LLM_MAX_TOKENS,
    )
    return response.choices[0].message.content.strip()


def parse_json_array(raw: str) -> list[dict]:
    """
    Robustly extract a JSON array from an LLM response that may be wrapped
    in markdown fences or have trailing commentary.
    """
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    start = cleaned.find("[")
    end   = cleaned.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON array found in LLM output:\n{raw[:500]}")
    return json.loads(cleaned[start : end + 1])

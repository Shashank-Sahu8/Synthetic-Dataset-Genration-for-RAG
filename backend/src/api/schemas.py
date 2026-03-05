"""
Pydantic request / response schemas for all FastAPI endpoints.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Project
# ─────────────────────────────────────────────────────────────────────────────

class ProjectCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, example="My RAG Benchmark Project")


class ProjectCreateResponse(BaseModel):
    project_id: str
    name: str
    api_key: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ProjectDetailResponse(BaseModel):
    project_id: str
    name: str
    created_at: datetime

    model_config = {"from_attributes": True}


# ─────────────────────────────────────────────────────────────────────────────
# Document Ingest
# ─────────────────────────────────────────────────────────────────────────────

class PageInput(BaseModel):
    """A single page of text as it arrives from the SDK / user."""
    page_no: int = Field(..., ge=1, example=1)
    text: str = Field(..., min_length=1)


class IngestRequest(BaseModel):
    """
    Body sent by the SDK to upload a document.
    `doc_id` is the human-readable identifier (e.g. "DOC-A").
    """
    doc_id: str = Field(..., min_length=1, max_length=255, example="DOC-A")
    pages: list[PageInput] = Field(..., min_length=1)
    # Pipeline parameters (optional – sensible defaults)
    batch_size: int = Field(5, ge=2, le=20)
    overlap: int = Field(1, ge=0, le=5)


class IngestResponse(BaseModel):
    message: str
    document_id: str
    project_id: str
    total_pages: int
    job_status: str = "processing"   # background job has been triggered


# ─────────────────────────────────────────────────────────────────────────────
# Batch (read-only, for monitoring)
# ─────────────────────────────────────────────────────────────────────────────

class BatchResponse(BaseModel):
    batch_id: str
    document_id: str
    batch_index: int
    page_ids: list[str]
    batch_context: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Entry (read-only)
# ─────────────────────────────────────────────────────────────────────────────

class DatasetEntryResponse(BaseModel):
    entry_id: str
    batch_id: str
    question: str
    answer: str
    source_context: str
    source_page_numbers: list[int]
    evaluation_scores: Optional[dict[str, Any]]
    overall_accuracy: Optional[float]
    is_faulty: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class DatasetListResponse(BaseModel):
    project_id: str
    total: int
    entries: list[DatasetEntryResponse]


# ─────────────────────────────────────────────────────────────────────────────
# Common error shape
# ─────────────────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    detail: str

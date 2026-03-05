"""
/dataset  –  retrieve generated QA entries for a project.

Authentication: X-API-Key header (same key the SDK uses).
"""
from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from sqlalchemy.orm import Session

from backend.database import get_db, Project, DatasetEntry, Batch
from backend.src.api.schemas import DatasetEntryResponse, DatasetListResponse

router = APIRouter(prefix="/dataset", tags=["Dataset"])


def _authenticate(api_key: str, db: Session) -> Project:
    project = db.query(Project).filter(Project.api_key == api_key).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )
    return project


@router.get(
    "/",
    response_model=DatasetListResponse,
    summary="Fetch all validated QA entries for your project",
)
def get_dataset(
    x_api_key: str = Header(..., alias="X-API-Key"),
    include_faulty: bool = Query(False, description="Include faulty entries in response"),
    db: Session = Depends(get_db),
):
    project = _authenticate(x_api_key, db)

    query = db.query(DatasetEntry).filter(DatasetEntry.project_id == project.id)
    if not include_faulty:
        query = query.filter(DatasetEntry.is_faulty == False)

    entries = query.order_by(DatasetEntry.created_at).all()

    return DatasetListResponse(
        project_id=str(project.id),
        total=len(entries),
        entries=[
            DatasetEntryResponse(
                entry_id=str(e.id),
                batch_id=str(e.batch_id),
                question=e.question,
                answer=e.answer,
                source_context=e.source_context,
                source_page_numbers=e.source_page_numbers or [],
                evaluation_scores=e.evaluation_scores,
                overall_accuracy=e.overall_accuracy,
                is_faulty=e.is_faulty,
                created_at=e.created_at,
            )
            for e in entries
        ],
    )


@router.get(
    "/status",
    summary="Get batch processing status for a document",
)
def get_batch_status(
    doc_id: str = Query(..., description="Human-readable doc_id, e.g. DOC-A"),
    x_api_key: str = Header(..., alias="X-API-Key"),
    db: Session = Depends(get_db),
):
    """Returns how many batches have been created and their contexts."""
    project = _authenticate(x_api_key, db)

    batches = (
        db.query(Batch)
        .filter(Batch.project_id == project.id)
        .order_by(Batch.batch_index)
        .all()
    )

    return {
        "project_id": str(project.id),
        "doc_id": doc_id,
        "total_batches": len(batches),
        "batches": [
            {
                "batch_index": b.batch_index,
                "batch_id": str(b.id),
                "has_context": bool(b.batch_context),
                "created_at": b.created_at.isoformat(),
            }
            for b in batches
        ],
    }

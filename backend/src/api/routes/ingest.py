"""
/ingest  –  document upload endpoint.

Flow
----
1. Authenticate request via X-API-Key header → resolve Project.
2. Persist Document + Page rows.
3. Launch Phase-1 LangGraph batch pipeline in a background thread.
4. Return immediately with document_id and job_status="processing".
"""
from __future__ import annotations

import uuid
import logging
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from backend.database import get_db, Project, Document, Page
from backend.src.api.schemas import IngestRequest, IngestResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["Ingest"])

# One shared thread-pool for background pipeline runs
_executor = ThreadPoolExecutor(max_workers=4)


def _authenticate(api_key: str, db: Session) -> Project:
    """Look up a Project by its API key.  Raises 401 if not found."""
    project = db.query(Project).filter(Project.api_key == api_key).first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key. Create a project first.",
        )
    return project


@router.post(
    "/",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a document and trigger the full pipeline (Phase 1-4)",
)
def ingest_document(
    body: IngestRequest,
    x_api_key: str = Header(..., alias="X-API-Key", description="Your project API key"),
    db: Session = Depends(get_db),
):
    # 1. Auth
    project = _authenticate(x_api_key, db)

    # 2. Persist Document
    doc = Document(
        id=uuid.uuid4(),
        project_id=project.id,
        doc_id=body.doc_id,
    )
    db.add(doc)
    db.flush()  # get doc.id before adding pages

    # 3. Persist Pages (ordered by page_no)
    sorted_pages = sorted(body.pages, key=lambda p: p.page_no)
    page_rows: list[Page] = []
    for p in sorted_pages:
        page_row = Page(
            id=uuid.uuid4(),
            document_id=doc.id,
            page_no=p.page_no,
            text=p.text,
        )
        db.add(page_row)
        page_rows.append(page_row)

    db.commit()

    # 4. Fire-and-forget: run full LangGraph pipeline (Phases 1-4) in background thread
    _executor.submit(
        _run_pipeline,
        str(project.id),
        str(doc.id),
        body.doc_id,
        [{"page_no": pr.page_no, "text": pr.text} for pr in page_rows],
        body.batch_size,
        body.overlap,
    )

    logger.info(
        "Ingest accepted  project=%s  doc=%s  pages=%d",
        project.id,
        doc.id,
        len(page_rows),
    )

    return IngestResponse(
        message="Document accepted. Full pipeline (batch → QA generation → RAGAS evaluation → persist) started in background.",
        document_id=str(doc.id),
        project_id=str(project.id),
        total_pages=len(page_rows),
    )


# ---------------------------------------------------------------------------
# Background pipeline runner
# ---------------------------------------------------------------------------

def _run_pipeline(
    project_id: str,
    document_id: str,
    doc_id: str,
    pages: list[dict],
    batch_size: int,
    overlap: int,
) -> None:
    """
    Executed in a thread-pool worker.
    Imports here to avoid circular imports at module load time.
    """
    try:
        from backend.src.graph.workflow import run_full_pipeline
        run_full_pipeline(
            project_id=project_id,
            document_id=document_id,
            doc_id=doc_id,
            pages=pages,
            batch_size=batch_size,
            overlap=overlap,
        )
        logger.info("Full pipeline completed  doc_id=%s", doc_id)
    except Exception:
        logger.exception("Full pipeline FAILED  doc_id=%s", doc_id)

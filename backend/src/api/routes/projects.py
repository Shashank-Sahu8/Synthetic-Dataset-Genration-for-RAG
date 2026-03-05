"""
/projects  –  project management endpoints.
"""
import secrets
import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.database import get_db, Project
from backend.src.api.schemas import (
    ProjectCreateRequest,
    ProjectCreateResponse,
    ProjectDetailResponse,
)

router = APIRouter(prefix="/projects", tags=["Projects"])


def _generate_api_key() -> str:
    """Generate a URL-safe, cryptographically secure API key."""
    return "sdk-" + secrets.token_urlsafe(32)


@router.post(
    "/",
    response_model=ProjectCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new project and receive a unique API key",
)
def create_project(
    body: ProjectCreateRequest,
    db: Session = Depends(get_db),
):
    existing = db.query(Project).filter(Project.name == body.name).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A project named '{body.name}' already exists.",
        )

    api_key = _generate_api_key()

    project = Project(
        id=uuid.uuid4(),
        name=body.name,
        api_key=api_key,
    )
    db.add(project)
    db.commit()
    db.refresh(project)

    return ProjectCreateResponse(
        project_id=str(project.id),
        name=project.name,
        api_key=project.api_key,
        created_at=project.created_at,
    )


@router.get(
    "/{project_id}",
    response_model=ProjectDetailResponse,
    summary="Get project metadata by project_id",
)
def get_project(project_id: str, db: Session = Depends(get_db)):
    try:
        pid = uuid.UUID(project_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid project_id format.")

    project = db.query(Project).filter(Project.id == pid).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found.")

    return ProjectDetailResponse(
        project_id=str(project.id),
        name=project.name,
        created_at=project.created_at,
    )

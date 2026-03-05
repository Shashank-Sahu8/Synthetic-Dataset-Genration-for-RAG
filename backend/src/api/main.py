"""
FastAPI application entry point.

Run with:
    uvicorn backend.src.api.main:app --reload --port 8000
"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.src.api.routes import projects, ingest, dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

app = FastAPI(
    title="Synthetic Dataset Generation API",
    description=(
        "SaaS platform for generating high-accuracy synthetic QA datasets "
        "from raw documents using LangGraph + RAGAS."
    ),
    version="1.0.0",
)

# Allow the Streamlit frontend (running on localhost during dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(projects.router)
app.include_router(ingest.router)
app.include_router(dataset.router)


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok"}

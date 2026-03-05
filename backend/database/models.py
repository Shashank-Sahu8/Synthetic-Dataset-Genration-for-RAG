import uuid
from datetime import datetime
from sqlalchemy import Column, String, Text, Integer, Boolean, ForeignKey, DateTime, Float
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Project(Base):
    """
    SaaS Tenant / Project Model.
    The SDK uniquely uses `api_key` to authenticate and identify where to store the data.
    """
    __tablename__ = "projects"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    api_key = Column(String(255), unique=True, index=True, nullable=False) # SDK Auth Key
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    documents = relationship("Document", back_populates="project", cascade="all, delete-orphan")
    batches = relationship("Batch", back_populates="project", cascade="all, delete-orphan")
    datasets = relationship("DatasetEntry", back_populates="project", cascade="all, delete-orphan")

class Document(Base):
    """
    Represents the uploaded JSON documentation (e.g., DOC-A, DOC-B).
    """
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    doc_id = Column(String(255), nullable=False, index=True) # Identifier from the user's JSON (e.g., "DOC-A")
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    project = relationship("Project", back_populates="documents")
    pages = relationship("Page", back_populates="document", cascade="all, delete-orphan")

class Page(Base):
    """
    Granular text chunks per document.
    """
    __tablename__ = "pages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    page_no = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)

    # Relationships
    document = relationship("Document", back_populates="pages")

class Batch(Base):
    """
    Stores overlapping batches of pages and their generated context to be used in QA generation.
    """
    __tablename__ = "batches"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    
    batch_index = Column(Integer, nullable=False)
    page_ids = Column(ARRAY(UUID(as_uuid=True)), nullable=False) # Array of Page IDs processed in this batch
    
    batch_context = Column(Text, nullable=True) # LLM generated summary (topics, entities, facts)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    project = relationship("Project", back_populates="batches")
    dataset_entries = relationship("DatasetEntry", back_populates="batch", cascade="all, delete-orphan")

class DatasetEntry(Base):
    """
    The generated and evaluated synthetic QA dataset pairs.
    """
    __tablename__ = "dataset_entries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    batch_id = Column(UUID(as_uuid=True), ForeignKey("batches.id", ondelete="CASCADE"), nullable=False)
    
    # Generated content
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    source_context = Column(Text, nullable=False)
    source_page_numbers = Column(ARRAY(Integer), nullable=False) # Array of page numbers serving as ground truth

    # RAGAS Evaluation metrics (JSON dict holding faithfulness, answer correctness, etc.)
    evaluation_scores = Column(JSONB, nullable=True)
    overall_accuracy = Column(Float, nullable=True) # E.g., Combined RAGAS score out of 1.0 (or 100)
    
    # Logical check state
    is_faulty = Column(Boolean, default=False) # Marked True if Accuracy < 80%

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    project = relationship("Project", back_populates="datasets")
    batch = relationship("Batch", back_populates="dataset_entries")

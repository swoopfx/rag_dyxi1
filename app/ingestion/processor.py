"""Production End-to-End Ingestion Processor for Enterprise Agentic RAG.

Architecture & Scope:
- Summary-Indexed Multi-Vector Architecture:
  * Embedded summaries of structured tables, images, and LaTeX-formatted formulas are
    stored directly in the Qdrant vector database for high-precision semantic matching.
  * Complete, intact, uncorrupted raw tables (full Markdown, raw JSON data rows, and HTML),
    images, and LaTeX formulas are persisted in MySQL (with SQLite resilience fallback),
    indexed by unique deterministic identifiers (`table_id`, `image_id`, `formula_id`).
- Query-Time Multi-Vector Table & Formula Retrieval:
  * At query time, vector search matches the user's question to the table summary,
    image summary, formula summary, or text passage.
  * Grabs `table_id` / `formula_id` from the match metadata.
  * Fetches the exact, uncorrupted, intact table or formula from MySQL to pass to the LLM.
- Composable LangChain LCEL Workflow:
  * Chained runnables: load -> chunk -> embed (summaries + text) -> store (Qdrant + MySQL).
- Fallout / Fallback Model Integration:
  * Primary: Google Gemini Embedding 2 ('gemini-embedding-2')
  * Fallout: Jina AI Multimodal ('jina-embeddings-v4')
  * Automatic per-batch failover or forced via prefer_fallback / USE_FALLOUT_MODEL.
- Logfire Distributed Observability:
  * Instrumenting every span with service name 'rag-ingestion-service'.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union
import uuid

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
from pydantic import BaseModel, Field

# SQLAlchemy imports for MySQL
import sqlalchemy
from sqlalchemy import BigInteger, Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Qdrant Vector DB imports
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.models import Distance, PointStruct, VectorParams

# LangChain LCEL imports
from langchain_core.runnables import RunnableLambda

# Project internal imports
from app.config import settings
from app.ingestion.chunking.chunker import (
    DEFAULT_MAX_TOKENS,
    ProcessedChunk,
    chunk_document,
)
from app.ingestion.loader.load import (
    ExtractedFormula,
    ExtractedImage,
    ExtractedTable,
    MultimodalDocument,
    load_file,
    scan_directory,
)
from app.services.retreival.embeddings import (
    DEFAULT_DIMENSION,
    DEFAULT_FALLBACK_MODEL,
    DEFAULT_PRIMARY_MODEL,
    DOCUMENT_TASK_TYPE,
    QUERY_TASK_TYPE,
    GeminiMultimodalEmbeddings,
    embed_query,
    embed_texts,
)

# ==============================================================================
# LOGFIRE OBSERVABILITY CONFIGURATION (INGESTION SERVICE)
# ==============================================================================

# LangChain tracing suppression when API key is missing
if not os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

INGESTION_SERVICE_NAME = getattr(
    settings, "LOGFIRE_INGESTION_SERVICE_NAME", "rag-ingestion-service"
)

try:
    import logfire

    _logfire_token = getattr(settings, "LOGFIRE_TOKEN", None) or os.getenv("LOGFIRE_TOKEN")
    logfire.configure(
        send_to_logfire="if-token-present",
        token=_logfire_token,
        service_name=INGESTION_SERVICE_NAME,
    )
except Exception:
    logfire = None


class _NoopSpan:
    """No-op span placeholder for when Logfire is unavailable."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_attributes(self, attributes: Dict[str, Any]) -> None:
        pass

    def record_exception(self, exception: BaseException) -> None:
        pass


_NOOP_SPAN = _NoopSpan()


@contextlib.contextmanager
def _logfire_span(name: str, **kwargs: Any):
    """Context manager for Logfire distributed tracing and span telemetry."""
    if logfire is not None:
        try:
            cm = logfire.span(name, **kwargs)
        except Exception:
            cm = None

        if cm is not None:
            with cm as span:
                yield span
            return

    yield _NOOP_SPAN


# ==============================================================================
# SQLALCHEMY RELATIONAL MODELS FOR INTACT DATA (MYSQL)
# ==============================================================================

Base = declarative_base()


class DocumentMetadataRecord(Base):
    """SQL table storing root document metadata and ingestion statistics."""

    __tablename__ = settings.TABLE_NAME_DOCUMENTS

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(String(128), unique=True, index=True, nullable=False)
    file_path = Column(String(1024), nullable=False)
    file_name = Column(String(255), nullable=False, index=True)
    file_extension = Column(String(32), nullable=False)
    file_size_bytes = Column(BigInteger, nullable=True)
    mime_type = Column(String(128), nullable=True)
    num_pages = Column(Integer, default=1)
    tables_count = Column(Integer, default=0)
    images_count = Column(Integer, default=0)
    formulas_count = Column(Integer, default=0)
    extra_metadata_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)


class ExtractedTableRecord(Base):
    """SQL lookup table storing complete, intact structured tables indexed by table_id.

    Stores the full, uncorrupted Markdown, HTML representation, raw rows JSON,
    and the summary that was embedded into Qdrant for semantic search matching.
    """

    __tablename__ = settings.TABLE_NAME_TABLES

    id = Column(Integer, primary_key=True, autoincrement=True)
    table_id = Column(String(128), unique=True, index=True, nullable=False)
    document_id = Column(String(128), index=True, nullable=False)
    table_index = Column(Integer, nullable=False)
    page_no = Column(Integer, nullable=True)
    caption = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)  # Embedded in Qdrant
    markdown = Column(Text, nullable=False)  # Full intact Markdown table
    html = Column(Text, nullable=True)  # Full intact HTML table
    data_rows_json = Column(Text, nullable=True)  # Full intact raw rows & headers JSON
    num_rows = Column(Integer, default=0)
    num_cols = Column(Integer, default=0)
    bbox_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)


class ExtractedImageRecord(Base):
    """SQL lookup table storing non-textual visual data indexed by image_id."""

    __tablename__ = settings.TABLE_NAME_IMAGES

    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(String(128), unique=True, index=True, nullable=False)
    document_id = Column(String(128), index=True, nullable=False)
    image_index = Column(Integer, nullable=False)
    page_no = Column(Integer, nullable=True)
    caption = Column(Text, nullable=True)
    description = Column(Text, nullable=True)  # AI-generated visual description from Gemini Flash Vision
    summary = Column(Text, nullable=True)  # Embedded in Qdrant
    label = Column(String(64), default="picture", nullable=False)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    format = Column(String(32), default="PNG", nullable=False)
    saved_path = Column(String(1024), nullable=True)
    base64_uri = Column(Text, nullable=True)
    bbox_json = Column(Text, nullable=True)
    annotations_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)


class ExtractedFormulaRecord(Base):
    """SQL lookup table storing LaTeX-formatted formulas indexed by formula_id."""

    __tablename__ = settings.TABLE_NAME_FORMULAS

    id = Column(Integer, primary_key=True, autoincrement=True)
    formula_id = Column(String(128), unique=True, index=True, nullable=False)
    document_id = Column(String(128), index=True, nullable=False)
    formula_index = Column(Integer, nullable=False)
    page_no = Column(Integer, nullable=True)
    latex = Column(Text, nullable=False)  # Full raw LaTeX formula string
    context = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)  # Embedded in Qdrant
    bbox_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)


# ==============================================================================
# DATABASE MANAGERS: MYSQL (NON-TEXTUAL) & QDRANT (VECTORS)
# ==============================================================================


class MySQLMetadataManager:
    """Production manager for persisting complete, uncorrupted non-textual data in MySQL.

    Connects to MySQL using PyMySQL driver via settings.get_mysql_url().
    Includes automatic resilience fallback to SQLite if the MySQL server is unreachable,
    guaranteeing seamless continuous operation in development and CI environments.
    """

    def __init__(self, database_url: Optional[str] = None):
        self.mysql_url = database_url or settings.get_mysql_url()
        self.sqlite_url = getattr(
            settings, "SQLITE_FALLBACK_URL", "sqlite:///rag_nontextual_metadata.db"
        )
        self.engine = None
        self.is_sqlite_fallback = False
        self._init_engine()
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False)

    def _init_engine(self) -> None:
        """Initialize engine, test connection, and create tables."""
        # 1. Attempt MySQL connection
        try:
            logger.info(f"Connecting to MySQL metadata store: {self.mysql_url}")
            engine = create_engine(
                self.mysql_url,
                pool_pre_ping=True,
                pool_recycle=3600,
            )
            with engine.connect() as conn:
                conn.execute(sqlalchemy.text("SELECT 1"))
            self.engine = engine
            self.is_sqlite_fallback = False
            if logfire is not None:
                try:
                    logfire.instrument_sqlalchemy(self.engine)
                except Exception:
                    pass
            logger.info("Successfully connected to MySQL database.")
        except Exception as e:
            logger.warning(
                f"Could not connect to MySQL database ({e}). "
                f"Falling back to local SQLite metadata store: '{self.sqlite_url}'"
            )
            self.engine = create_engine(self.sqlite_url)
            self.is_sqlite_fallback = True
            if logfire is not None:
                try:
                    logfire.instrument_sqlalchemy(self.engine)
                except Exception:
                    pass

        # 2. Ensure schema tables exist
        try:
            Base.metadata.create_all(self.engine)
            logger.info("Non-textual metadata schema tables verified/created successfully.")
        except Exception as schema_err:
            logger.error(f"Error creating metadata schema tables: {schema_err}")
            raise

    @contextlib.contextmanager
    def get_session(self):
        """Context manager providing transactional session scope."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def store_document_nontextual_data(self, doc: MultimodalDocument) -> Dict[str, Any]:
        """Store document metadata, intact tables, images, and formulas in MySQL.

        Args:
            doc: MultimodalDocument instance returned by load_file.

        Returns:
            Dictionary summarizing stored record IDs and counts.
        """
        doc_path = doc.file_path or doc.file_name
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(Path(doc_path).resolve())))

        with _logfire_span(
            "mysql.store_nontextual",
            document_id=doc_id,
            file_name=doc.file_name,
            tables_count=len(doc.tables),
            images_count=len(doc.images),
            formulas_count=len(doc.formulas),
        ):
            with self.get_session() as session:
                # 1. Upsert / update root DocumentMetadataRecord
                existing_doc = (
                    session.query(DocumentMetadataRecord)
                    .filter(DocumentMetadataRecord.document_id == doc_id)
                    .first()
                )
                if existing_doc:
                    existing_doc.tables_count = len(doc.tables)
                    existing_doc.images_count = len(doc.images)
                    existing_doc.formulas_count = len(doc.formulas)
                    existing_doc.num_pages = doc.num_pages
                    existing_doc.extra_metadata_json = json.dumps(doc.metadata)
                    # Clear out previous records to avoid duplicate IDs on re-ingestion
                    session.query(ExtractedTableRecord).filter(
                        ExtractedTableRecord.document_id == doc_id
                    ).delete()
                    session.query(ExtractedImageRecord).filter(
                        ExtractedImageRecord.document_id == doc_id
                    ).delete()
                    session.query(ExtractedFormulaRecord).filter(
                        ExtractedFormulaRecord.document_id == doc_id
                    ).delete()
                else:
                    new_doc = DocumentMetadataRecord(
                        document_id=doc_id,
                        file_path=str(doc.file_path) if doc.file_path else doc.file_name,
                        file_name=doc.file_name,
                        file_extension=doc.file_extension,
                        file_size_bytes=doc.file_size_bytes,
                        mime_type=doc.mime_type,
                        num_pages=doc.num_pages,
                        tables_count=len(doc.tables),
                        images_count=len(doc.images),
                        formulas_count=len(doc.formulas),
                        extra_metadata_json=json.dumps(doc.metadata),
                    )
                    session.add(new_doc)

                # 2. Store Complete, Intact Tables (indexed by table_id)
                for tbl in doc.tables:
                    table_id = f"{doc_id}#table_{tbl.index:03d}"
                    summary = tbl.generate_summary()
                    bbox_data = tbl.bbox.model_dump() if tbl.bbox else None
                    tbl_record = ExtractedTableRecord(
                        table_id=table_id,
                        document_id=doc_id,
                        table_index=tbl.index,
                        page_no=tbl.page_no,
                        caption=tbl.caption,
                        summary=summary,
                        markdown=tbl.markdown,  # Complete, intact Markdown table
                        html=tbl.html,
                        data_rows_json=json.dumps(tbl.data_rows) if tbl.data_rows else None,
                        num_rows=tbl.num_rows,
                        num_cols=tbl.num_cols,
                        bbox_json=json.dumps(bbox_data) if bbox_data else None,
                    )
                    session.add(tbl_record)

                # 3. Store Images / Figures / Charts (indexed by image_id)
                for img in doc.images:
                    image_id = f"{doc_id}#image_{img.index:03d}"
                    summary = img.generate_summary(use_vision=True)
                    bbox_data = img.bbox.model_dump() if img.bbox else None
                    img_record = ExtractedImageRecord(
                        image_id=image_id,
                        document_id=doc_id,
                        image_index=img.index,
                        page_no=img.page_no,
                        caption=img.caption,
                        description=getattr(img, "description", None),
                        summary=summary,
                        label=img.label or "picture",
                        width=img.width,
                        height=img.height,
                        format=img.format,
                        saved_path=img.saved_path,
                        base64_uri=img.base64_uri,
                        bbox_json=json.dumps(bbox_data) if bbox_data else None,
                        annotations_json=json.dumps(img.annotations) if img.annotations else None,
                    )
                    session.add(img_record)

                # 4. Store LaTeX-formatted Formulas (indexed by formula_id)
                for f in doc.formulas:
                    formula_id = f"{doc_id}#formula_{f.index:03d}"
                    summary = f.generate_summary()
                    bbox_data = f.bbox.model_dump() if f.bbox else None
                    f_record = ExtractedFormulaRecord(
                        formula_id=formula_id,
                        document_id=doc_id,
                        formula_index=f.index,
                        page_no=f.page_no,
                        latex=f.latex,  # Complete raw LaTeX formula
                        context=f.context,
                        summary=summary,
                        bbox_json=json.dumps(bbox_data) if bbox_data else None,
                    )
                    session.add(f_record)

            logger.info(
                f"MySQL stored non-textual content for '{doc.file_name}': "
                f"{len(doc.tables)} tables, {len(doc.images)} images, "
                f"{len(doc.formulas)} formulas (doc_id={doc_id[:8]}...)."
            )
            return {
                "document_id": doc_id,
                "tables_count": len(doc.tables),
                "images_count": len(doc.images),
                "formulas_count": len(doc.formulas),
                "db_engine": "sqlite" if self.is_sqlite_fallback else "mysql",
            }

    def get_table_by_id(self, table_id: str) -> Optional[Dict[str, Any]]:
        """Fetch the exact, uncorrupted, intact table from MySQL by table_id."""
        with self.get_session() as session:
            record = (
                session.query(ExtractedTableRecord)
                .filter(ExtractedTableRecord.table_id == table_id)
                .first()
            )
            if not record:
                return None
            return {
                "table_id": record.table_id,
                "document_id": record.document_id,
                "caption": record.caption,
                "summary": record.summary,
                "markdown": record.markdown,  # Complete intact Markdown table
                "html": record.html,
                "data_rows": json.loads(record.data_rows_json) if record.data_rows_json else [],
                "num_rows": record.num_rows,
                "num_cols": record.num_cols,
                "page_no": record.page_no,
            }

    def get_formula_by_id(self, formula_id: str) -> Optional[Dict[str, Any]]:
        """Fetch the exact LaTeX formula from MySQL by formula_id."""
        with self.get_session() as session:
            record = (
                session.query(ExtractedFormulaRecord)
                .filter(ExtractedFormulaRecord.formula_id == formula_id)
                .first()
            )
            if not record:
                return None
            return {
                "formula_id": record.formula_id,
                "document_id": record.document_id,
                "latex": record.latex,  # Exact raw LaTeX string
                "context": record.context,
                "summary": record.summary,
                "page_no": record.page_no,
            }

    def get_image_by_id(self, image_id: str) -> Optional[Dict[str, Any]]:
        """Fetch the image record from MySQL by image_id."""
        with self.get_session() as session:
            record = (
                session.query(ExtractedImageRecord)
                .filter(ExtractedImageRecord.image_id == image_id)
                .first()
            )
            if not record:
                return None
            return {
                "image_id": record.image_id,
                "document_id": record.document_id,
                "caption": record.caption,
                "description": record.description,
                "summary": record.summary,
                "label": record.label,
                "format": record.format,
                "saved_path": record.saved_path,
                "base64_uri": record.base64_uri,
                "page_no": record.page_no,
            }

    def wipe_all_data(self) -> Dict[str, int]:
        """Wipe all extracted tables, formulas, images, and document metadata from MySQL/SQLite.

        Returns:
            Dictionary with deletion counts per table.
        """
        with _logfire_span("mysql.wipe_all_data"):
            with self.get_session() as session:
                num_tables = session.query(ExtractedTableRecord).delete()
                num_formulas = session.query(ExtractedFormulaRecord).delete()
                num_images = session.query(ExtractedImageRecord).delete()
                num_docs = session.query(DocumentMetadataRecord).delete()
            logger.warning(
                f"Wiped all non-textual data from relational database: {num_docs} document(s), "
                f"{num_tables} table(s), {num_formulas} formula(s), {num_images} image(s)."
            )
            return {
                "documents_deleted": num_docs,
                "tables_deleted": num_tables,
                "formulas_deleted": num_formulas,
                "images_deleted": num_images,
            }


class VectorIndexEntry(BaseModel):
    """An individual entry to be vectorized and indexed in Qdrant."""

    entry_id: str
    content_type: str  # "text" | "table_summary" | "image_summary" | "formula_summary"
    text_to_embed: str
    payload: Dict[str, Any]


class QdrantVectorManager:
    """Production manager for Qdrant vector database operations.

    Connects to cloud Qdrant cluster via settings.QDRANT_CLUSTER_ENDPOINT and
    settings.QDRANT_API_KEY, or local persistent storage if credentials are unset.
    Ensures collection initialization with proper vector dimensionality and Cosine distance.
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        default_collection: Optional[str] = None,
    ):
        self.endpoint = endpoint or settings.QDRANT_CLUSTER_ENDPOINT
        self.api_key = api_key or settings.QDRANT_API_KEY
        self.collection_name = (
            default_collection
            or getattr(settings, "QDRANT_COLLECTION", "dyxi_stag")
            or "dyxi_stag"
        )
        self.client = self._init_client()

    def _init_client(self) -> QdrantClient:
        """Initialize Qdrant client connection with cloud REST and local storage fallback."""
        if self.endpoint:
            logger.info(f"Connecting to Qdrant cluster at {self.endpoint} (REST mode)...")
            try:
                client = QdrantClient(
                    url=self.endpoint,
                    api_key=self.api_key,
                    prefer_grpc=False,
                    timeout=25.0,
                )
                # Verify cluster connection
                client.get_collections()
                logger.info("Successfully connected to Qdrant cluster.")
                return client
            except Exception as e:
                logger.warning(
                    f"Could not connect to Qdrant cluster ({e}). "
                    "Falling back to local persistent Qdrant at './qdrant_storage'..."
                )
                return QdrantClient(path="./qdrant_storage")
        else:
            local_path = "./qdrant_storage"
            logger.info(f"No QDRANT_CLUSTER_ENDPOINT set. Initializing local Qdrant at '{local_path}'")
            return QdrantClient(path=local_path)

    def ensure_collection(
        self,
        collection_name: Optional[str] = None,
        vector_size: int = DEFAULT_DIMENSION,
        distance: Distance = Distance.COSINE,
    ) -> str:
        """Ensure the target Qdrant collection exists with the required dimensionality."""
        col = collection_name or self.collection_name
        with _logfire_span("qdrant.ensure_collection", collection=col, vector_size=vector_size):
            try:
                collections = [c.name for c in self.client.get_collections().collections]
                if col not in collections:
                    logger.info(
                        f"Creating Qdrant collection '{col}' with dim={vector_size}, distance={distance}..."
                    )
                    self.client.create_collection(
                        collection_name=col,
                        vectors_config=VectorParams(size=vector_size, distance=distance),
                    )
                return col
            except Exception as e:
                logger.error(f"Failed to verify/create Qdrant collection '{col}': {e}")
                raise

    def upsert_entries(
        self,
        entries: Sequence[VectorIndexEntry],
        embeddings: Sequence[Sequence[float]],
        collection_name: Optional[str] = None,
        model_name: Optional[str] = None,
        batch_size: int = 64,
    ) -> int:
        """Upsert vector entries (text chunks, table summaries, formula summaries) into Qdrant.

        Args:
            entries: List of VectorIndexEntry instances.
            embeddings: Corresponding embedding vectors.
            collection_name: Target Qdrant collection.
            model_name: Name of model that generated the embeddings.
            batch_size: Batch size for point upserts.

        Returns:
            Count of points upserted.
        """
        if not entries or not embeddings:
            return 0

        if len(entries) != len(embeddings):
            raise ValueError(
                f"Entries count ({len(entries)}) does not match embeddings count ({len(embeddings)})"
            )

        col = collection_name or self.collection_name
        self.ensure_collection(col, vector_size=len(embeddings[0]))

        total_upserted = 0
        timestamp = datetime.datetime.utcnow().isoformat()

        with _logfire_span(
            "qdrant.upsert_entries",
            collection=col,
            total_points=len(entries),
            model=model_name or DEFAULT_PRIMARY_MODEL,
        ):
            points: List[PointStruct] = []
            for entry, vec in zip(entries, embeddings):
                # Deterministic UUID5 based on entry_id guarantees idempotence
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, entry.entry_id))
                payload = dict(entry.payload)
                payload["model_used"] = model_name or DEFAULT_PRIMARY_MODEL
                payload["timestamp"] = timestamp

                points.append(PointStruct(id=point_id, vector=list(vec), payload=payload))

            for i in range(0, len(points), batch_size):
                batch_points = points[i : i + batch_size]
                self.client.upsert(collection_name=col, points=batch_points)
                total_upserted += len(batch_points)

            logger.info(f"Upserted {total_upserted} vectors into Qdrant collection '{col}'.")
            return total_upserted

    def wipe_collection(
        self,
        collection_name: Optional[str] = None,
        vector_size: int = DEFAULT_DIMENSION,
        distance: Distance = Distance.COSINE,
    ) -> bool:
        """Delete and recreate target collection in Qdrant, completely wiping all points.

        Args:
            collection_name: Target Qdrant collection name (default: self.collection_name).
            vector_size: Dimensionality for recreated collection.
            distance: Vector distance metric.

        Returns:
            True if collection was wiped and recreated.
        """
        col = collection_name or self.collection_name
        with _logfire_span("qdrant.wipe_collection", collection=col):
            try:
                collections = [c.name for c in self.client.get_collections().collections]
                if col in collections:
                    logger.warning(f"Deleting existing Qdrant collection '{col}' for clean re-install...")
                    self.client.delete_collection(collection_name=col)
                # Re-create empty collection with proper configuration
                self.ensure_collection(collection_name=col, vector_size=vector_size, distance=distance)
                try:
                    col_info = self.client.get_collection(collection_name=col)
                    points_count = col_info.points_count or 0
                except Exception:
                    points_count = 0
                logger.info(f"Successfully wiped and re-created empty Qdrant collection '{col}' (points: {points_count}).")
                return True
            except Exception as e:
                logger.error(f"Failed to wipe/recreate Qdrant collection '{col}': {e}")
                raise


def wipe_all_storage(
    collection_name: Optional[str] = None,
    mysql_manager: Optional[MySQLMetadataManager] = None,
    qdrant_manager: Optional[QdrantVectorManager] = None,
) -> Dict[str, Any]:
    """Wipe all data across both Qdrant vector database and MySQL metadata database.

    Args:
        collection_name: Target Qdrant collection name to wipe.
        mysql_manager: Optional shared MySQL manager.
        qdrant_manager: Optional shared Qdrant manager.

    Returns:
        Dictionary summarizing deleted records across both stores.
    """
    db_mgr = mysql_manager or MySQLMetadataManager()
    qd_mgr = qdrant_manager or QdrantVectorManager(default_collection=collection_name)
    target_col = collection_name or qd_mgr.collection_name

    logger.warning(
        f"Initiating full wipe across Qdrant collection '{target_col}' and "
        "MySQL metadata store..."
    )
    with _logfire_span("storage.wipe_all", collection=target_col):
        # 1. Wipe MySQL / SQLite relational store
        mysql_stats = db_mgr.wipe_all_data()
        # 2. Wipe Qdrant vector collection
        qd_mgr.wipe_collection(collection_name=target_col)

        try:
            col_info = qd_mgr.client.get_collection(collection_name=target_col)
            qdrant_points = col_info.points_count or 0
        except Exception:
            qdrant_points = 0

        logger.info(
            f"Full storage wipe completed successfully: Qdrant collection '{target_col}' recreated (points: {qdrant_points}), "
            f"MySQL purged ({mysql_stats})."
        )
        return {
            "status": "success",
            "qdrant_collection_wiped": target_col,
            "qdrant_points_remaining": qdrant_points,
            "mysql_deleted": mysql_stats,
        }


# ==============================================================================
# PIPELINE DATA STRUCTURES & SUMMARY MODELS
# ==============================================================================


class FileIngestionSummary(BaseModel):
    """Summary metrics for a single ingested file."""

    file_path: str
    file_name: str
    document_id: str
    chunks_count: int
    tables_count: int
    images_count: int
    formulas_count: int
    vectors_count: int
    model_used: str
    db_engine: str
    time_taken_seconds: float
    total_tokens: int = Field(default=0, description="Total tokens consumed across all text chunks")
    avg_tokens_per_chunk: float = Field(default=0.0, description="Average tokens per chunk")
    total_embedded_tokens: int = Field(default=0, description="Total tokens embedded across all vectors")
    throughput_chunks_per_sec: float = Field(default=0.0, description="Chunk generation throughput")
    throughput_tokens_per_sec: float = Field(default=0.0, description="Token processing throughput")


class DirectoryIngestionSummary(BaseModel):
    """Aggregated summary metrics for recursive directory ingestion."""

    directory: str
    total_files_scanned: int
    successful_files: int
    failed_files: int
    total_chunks: int
    total_tables: int
    total_images: int
    total_formulas: int
    total_vectors: int
    model_used: str
    duration_seconds: float
    total_tokens: int = Field(default=0, description="Total tokens consumed across all files")
    avg_tokens_per_chunk: float = Field(default=0.0, description="Average tokens per chunk across all files")
    avg_tokens_per_file: float = Field(default=0.0, description="Average tokens per file")
    total_embedded_tokens: int = Field(default=0, description="Total tokens embedded across all vectors")
    throughput_chunks_per_sec: float = Field(default=0.0, description="Overall chunks processed per second")
    throughput_tokens_per_sec: float = Field(default=0.0, description="Overall tokens processed per second")
    file_summaries: List[FileIngestionSummary] = Field(default_factory=list)
    errors: Dict[str, str] = Field(default_factory=dict)


# ==============================================================================
# LANGCHAIN INGESTION WORKFLOW (SUMMARY-INDEXED RUNNABLES)
# ==============================================================================


class LangChainIngestionWorkflow:
    """Production LangChain LCEL workflow for Summary-Indexed document ingestion.

    Composes:
    1. DataLoaderRunnable: reads source file via load.py into MultimodalDocument
    2. ChunkerRunnable: chunks document via chunker.py into ProcessedChunks
    3. EmbeddingsRunnable: generates vectors for text chunks, table summaries,
       image summaries, and LaTeX formula summaries using embeddings.py
    4. StorageRunnable: stores full, uncorrupted tables and formulas into MySQL,
       and upserts all vectors into Qdrant with direct lookup IDs.
    """

    def __init__(
        self,
        mysql_manager: Optional[MySQLMetadataManager] = None,
        qdrant_manager: Optional[QdrantVectorManager] = None,
        prefer_fallback: bool = False,
        primary_model: str = DEFAULT_PRIMARY_MODEL,
        fallback_model: str = DEFAULT_FALLBACK_MODEL,
        dimension: int = DEFAULT_DIMENSION,
        collection_name: Optional[str] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self.mysql_manager = mysql_manager or MySQLMetadataManager()
        self.qdrant_manager = qdrant_manager or QdrantVectorManager()
        self.prefer_fallback = prefer_fallback or getattr(settings, "USE_FALLOUT_MODEL", False)
        self.primary_model = primary_model
        self.fallback_model = fallback_model
        self.dimension = dimension
        self.collection_name = collection_name or self.qdrant_manager.collection_name
        self.max_tokens = max_tokens

        # LangChain Embeddings adapter
        self.embeddings_adapter = GeminiMultimodalEmbeddings(
            model=self.primary_model,
            fallback_model=self.fallback_model,
            dimension=self.dimension,
            prefer_fallback=self.prefer_fallback,
        )

        # Build LCEL runnable sequence
        self.chain = self._build_chain()

    def _load_step(self, file_path_input: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        """Step 1: Load file using load.py into MultimodalDocument."""
        if isinstance(file_path_input, dict):
            p = file_path_input.get("file_path")
        else:
            p = file_path_input

        path_obj = Path(p).resolve()
        logger.debug(f"[LangChain Workflow] Step 1: Loading '{path_obj.name}'...")
        doc = load_file(path_obj)
        return {
            "file_path": str(path_obj),
            "document": doc,
            "start_time": time.time(),
        }

    def _chunk_step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Step 2: Structure-aware chunking using chunker.py."""
        doc: MultimodalDocument = state["document"]
        logger.debug(f"[LangChain Workflow] Step 2: Chunking '{doc.file_name}'...")
        chunks = chunk_document(
            document=doc,
            max_tokens=self.max_tokens,
            merge_peers=True,
        )
        total_tokens = sum(c.metadata.num_tokens or 0 for c in chunks)
        avg_tokens = round(total_tokens / len(chunks), 2) if chunks else 0.0
        state["chunks"] = chunks
        state["total_tokens"] = total_tokens
        state["avg_tokens_per_chunk"] = avg_tokens
        return state

    def _embed_step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Step 3: Multi-Vector Embedding of text chunks, table summaries, image summaries, and LaTeX formulas."""
        doc: MultimodalDocument = state["document"]
        chunks: List[ProcessedChunk] = state["chunks"]
        doc_path = doc.file_path or doc.file_name
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(Path(doc_path).resolve())))

        entries_to_index: List[VectorIndexEntry] = []

        # 1. Text Chunks
        for chunk in chunks:
            entries_to_index.append(
                VectorIndexEntry(
                    entry_id=chunk.chunk_id,
                    content_type="text",
                    text_to_embed=chunk.embedding_text,
                    payload={
                        "content_type": "text",
                        "chunk_id": chunk.chunk_id,
                        "document_id": doc_id,
                        "source_file": chunk.metadata.source_file,
                        "page_numbers": chunk.metadata.page_numbers,
                        "headings": chunk.metadata.headings,
                        "captions": chunk.metadata.captions,
                        "text": chunk.text,
                        "contextualized_text": chunk.contextualized_text,
                        "num_tokens": chunk.metadata.num_tokens,
                        "is_contextualized": chunk.metadata.is_contextualized,
                    },
                )
            )

        # 2. Table Summaries for Vector Search (linked to full intact tables in MySQL)
        for tbl in doc.tables:
            table_id = f"{doc_id}#table_{tbl.index:03d}"
            table_summary = tbl.generate_summary()
            entries_to_index.append(
                VectorIndexEntry(
                    entry_id=table_id,
                    content_type="table_summary",
                    text_to_embed=table_summary,
                    payload={
                        "content_type": "table_summary",
                        "table_id": table_id,  # KEY TO FETCH COMPLETE INTACT TABLE FROM MYSQL
                        "document_id": doc_id,
                        "source_file": doc.file_path or doc.file_name,
                        "page_numbers": [tbl.page_no] if tbl.page_no else [],
                        "summary": table_summary,
                        "caption": tbl.caption,
                        "num_rows": tbl.num_rows,
                        "num_cols": tbl.num_cols,
                    },
                )
            )

        # 3. Image Summaries for Vector Search (with Gemini Flash Vision description)
        for img in doc.images:
            image_id = f"{doc_id}#image_{img.index:03d}"
            image_summary = img.generate_summary(use_vision=True)
            entries_to_index.append(
                VectorIndexEntry(
                    entry_id=image_id,
                    content_type="image_summary",
                    text_to_embed=image_summary,
                    payload={
                        "content_type": "image_summary",
                        "image_id": image_id,  # KEY TO FETCH IMAGE METADATA FROM MYSQL
                        "document_id": doc_id,
                        "source_file": doc.file_path or doc.file_name,
                        "page_numbers": [img.page_no] if img.page_no else [],
                        "summary": image_summary,
                        "description": getattr(img, "description", None),
                        "caption": img.caption,
                        "label": img.label,
                        "saved_path": img.saved_path,
                    },
                )
            )

        # 4. LaTeX-Formatted Formula Summaries for Vector Search
        for f in doc.formulas:
            formula_id = f"{doc_id}#formula_{f.index:03d}"
            formula_summary = f.generate_summary()
            entries_to_index.append(
                VectorIndexEntry(
                    entry_id=formula_id,
                    content_type="formula_summary",
                    text_to_embed=formula_summary,
                    payload={
                        "content_type": "formula_summary",
                        "formula_id": formula_id,  # KEY TO FETCH EXACT LATEX FORMULA FROM MYSQL
                        "document_id": doc_id,
                        "source_file": doc.file_path or doc.file_name,
                        "page_numbers": [f.page_no] if f.page_no else [],
                        "summary": formula_summary,
                        "latex": f.latex,
                        "context": f.context,
                    },
                )
            )

        logger.debug(
            f"[LangChain Workflow] Step 3: Embedding {len(entries_to_index)} item(s) "
            f"({len(chunks)} chunks, {len(doc.tables)} table summaries, "
            f"{len(doc.images)} image summaries, {len(doc.formulas)} formula summaries) "
            f"(prefer_fallback={self.prefer_fallback})..."
        )

        if entries_to_index:
            texts = [entry.text_to_embed for entry in entries_to_index]
            vectors = embed_texts(
                texts=texts,
                model=self.primary_model,
                dimension=self.dimension,
                task_type=DOCUMENT_TASK_TYPE,
                fallback_model=self.fallback_model,
                prefer_fallback=self.prefer_fallback,
            )
        else:
            vectors = []

        from app.ingestion.chunking.chunker import GeminiTokenEstimator
        estimator = GeminiTokenEstimator(max_tokens=self.max_tokens)
        total_embedded_tokens = sum(estimator.count_tokens(entry.text_to_embed) for entry in entries_to_index)

        active_model = self.fallback_model if self.prefer_fallback else self.primary_model
        state["entries_to_index"] = entries_to_index
        state["embeddings"] = vectors
        state["model_used"] = active_model
        state["total_embedded_tokens"] = total_embedded_tokens
        return state

    def _store_step(self, state: Dict[str, Any]) -> FileIngestionSummary:
        """Step 4: Persist complete uncorrupted data into MySQL and summary vectors into Qdrant."""
        doc: MultimodalDocument = state["document"]
        chunks: List[ProcessedChunk] = state["chunks"]
        entries: List[VectorIndexEntry] = state["entries_to_index"]
        embeddings: List[List[float]] = state["embeddings"]
        model_used: str = state["model_used"]
        start_time: float = state["start_time"]
        total_tokens: int = state.get("total_tokens", 0)
        avg_tokens: float = state.get("avg_tokens_per_chunk", round(total_tokens / len(chunks), 2) if chunks else 0.0)
        total_embedded_tokens: int = state.get("total_embedded_tokens", 0)

        logger.debug(f"[LangChain Workflow] Step 4: Storing '{doc.file_name}'...")

        # 1. Store complete, intact structured tables, images, and LaTeX formulas in MySQL
        mysql_result = self.mysql_manager.store_document_nontextual_data(doc)

        # 2. Store embedded summaries and text vectors in Qdrant
        vectors_count = 0
        if entries and embeddings:
            vectors_count = self.qdrant_manager.upsert_entries(
                entries=entries,
                embeddings=embeddings,
                collection_name=self.collection_name,
                model_name=model_used,
            )

        elapsed = time.time() - start_time
        throughput_chunks = round(len(chunks) / elapsed, 2) if elapsed > 0 else 0.0
        throughput_tokens = round(total_tokens / elapsed, 2) if elapsed > 0 else 0.0

        return FileIngestionSummary(
            file_path=state["file_path"],
            file_name=doc.file_name,
            document_id=mysql_result["document_id"],
            chunks_count=len(chunks),
            tables_count=len(doc.tables),
            images_count=len(doc.images),
            formulas_count=len(doc.formulas),
            vectors_count=vectors_count,
            model_used=model_used,
            db_engine=mysql_result["db_engine"],
            time_taken_seconds=round(elapsed, 2),
            total_tokens=total_tokens,
            avg_tokens_per_chunk=avg_tokens,
            total_embedded_tokens=total_embedded_tokens,
            throughput_chunks_per_sec=throughput_chunks,
            throughput_tokens_per_sec=throughput_tokens,
        )

    def _build_chain(self):
        """Construct composable LCEL chain using RunnableLambda."""
        return (
            RunnableLambda(self._load_step)
            | RunnableLambda(self._chunk_step)
            | RunnableLambda(self._embed_step)
            | RunnableLambda(self._store_step)
        )

    def invoke(self, file_path: Union[str, Path]) -> FileIngestionSummary:
        """Execute end-to-end ingestion on a single file through the LangChain pipeline."""
        with _logfire_span("workflow.invoke", file_path=str(file_path)) as span:
            summary = self.chain.invoke(str(file_path))
            span.set_attributes({
                "ingestion.chunks_count": summary.chunks_count,
                "ingestion.total_tokens": summary.total_tokens,
                "ingestion.avg_tokens_per_chunk": summary.avg_tokens_per_chunk,
                "ingestion.total_embedded_tokens": summary.total_embedded_tokens,
                "ingestion.tables_count": summary.tables_count,
                "ingestion.images_count": summary.images_count,
                "ingestion.formulas_count": summary.formulas_count,
                "ingestion.vectors_count": summary.vectors_count,
                "ingestion.model_used": summary.model_used,
                "ingestion.db_engine": summary.db_engine,
                "ingestion.time_taken_seconds": summary.time_taken_seconds,
                "ingestion.throughput_chunks_per_sec": summary.throughput_chunks_per_sec,
                "ingestion.throughput_tokens_per_sec": summary.throughput_tokens_per_sec,
            })
            logger.info(
                f"[Logfire] Ingested '{summary.file_name}': {summary.chunks_count} chunks created, "
                f"{summary.total_tokens} tokens consumed (~{summary.avg_tokens_per_chunk:.1f} tok/chunk), "
                f"{summary.vectors_count} vectors indexed in Qdrant ({summary.model_used}), "
                f"{summary.tables_count} tables, {summary.images_count} images, {summary.formulas_count} formulas in MySQL "
                f"in {summary.time_taken_seconds}s ({summary.throughput_tokens_per_sec:.1f} tokens/s)."
            )
            return summary

    def batch(self, file_paths: Sequence[Union[str, Path]]) -> List[FileIngestionSummary]:
        """Execute batch ingestion across multiple files using the LangChain pipeline."""
        with _logfire_span("workflow.batch", total_files=len(file_paths)):
            return self.chain.batch([str(p) for p in file_paths])


# ==============================================================================
# MAIN INGESTION FUNCTIONS (BINDING THE ENTIRE PROJECT)
# ==============================================================================


def ingest_file(
    file_path: Union[str, Path],
    prefer_fallback: bool = False,
    collection_name: Optional[str] = None,
    mysql_manager: Optional[MySQLMetadataManager] = None,
    qdrant_manager: Optional[QdrantVectorManager] = None,
) -> FileIngestionSummary:
    """Ingest a single document into Qdrant (summary vectors) and MySQL (intact non-textual data)."""
    workflow = LangChainIngestionWorkflow(
        mysql_manager=mysql_manager,
        qdrant_manager=qdrant_manager,
        prefer_fallback=prefer_fallback,
        collection_name=collection_name,
    )
    return workflow.invoke(file_path)


def ingest_directory(
    directory_path: Union[str, Path] = "DATA",
    sub_dir: Optional[str] = None,
    max_files: Optional[int] = None,
    wipe: bool = False,
    recursive: bool = True,
    prefer_fallback: bool = False,
    collection_name: Optional[str] = None,
    ignore_errors: bool = True,
    mysql_manager: Optional[MySQLMetadataManager] = None,
    qdrant_manager: Optional[QdrantVectorManager] = None,
) -> DirectoryIngestionSummary:
    """Recursively read documents in directory (or specific sub-directory) and ingest into Qdrant & MySQL.

    Args:
        directory_path: Root directory containing documents (default: DATA).
        sub_dir: Optional specific subdirectory within directory_path (e.g. 'finetunes', 'noisy').
                 When provided, only files in this targeted subdirectory are scanned and ingested.
        max_files: Optional ceiling on the number of files to process.
        wipe: If True, completely wipes all data in the target Qdrant collection and MySQL metadata
              store before re-installing and ingesting the new set of data.
        recursive: Whether to scan subdirectories recursively.
        prefer_fallback: Whether to route embeddings to the fallback/fallout model.
        collection_name: Target Qdrant collection override (defaults to settings.QDRANT_COLLECTION or 'dyxi_stag').
        ignore_errors: Whether to continue processing remaining files if an individual file fails.
        mysql_manager: Optional shared MySQL manager.
        qdrant_manager: Optional shared Qdrant manager.

    Returns:
        DirectoryIngestionSummary summarizing counts of files, chunks, tables, formulas, and vectors.
    """
    base_dir = Path(directory_path).resolve()
    if sub_dir:
        target_dir = (base_dir / sub_dir).resolve()
        if not target_dir.exists():
            alt_path = Path(sub_dir).resolve()
            if alt_path.exists():
                target_dir = alt_path
            else:
                logger.warning(
                    f"Subdirectory '{sub_dir}' not found at '{target_dir}'. Scanning target may be empty."
                )
    else:
        target_dir = base_dir

    start_time = time.time()
    active_model = DEFAULT_FALLBACK_MODEL if prefer_fallback else DEFAULT_PRIMARY_MODEL

    db_mgr = mysql_manager or MySQLMetadataManager()
    qd_mgr = qdrant_manager or QdrantVectorManager(default_collection=collection_name)
    active_col = collection_name or qd_mgr.collection_name

    # 1. Execute Wipe if requested
    if wipe:
        logger.warning(
            f"[WIPE TRIGGERED] Purging all points in Qdrant collection '{active_col}' "
            "and all records in MySQL metadata tables..."
        )
        wipe_all_storage(
            collection_name=active_col,
            mysql_manager=db_mgr,
            qdrant_manager=qd_mgr,
        )

    logger.info(
        f"Starting directory ingestion on '{target_dir}' "
        f"(sub_dir='{sub_dir or 'None'}', max_files={max_files or 'All'}, wipe={wipe}, "
        f"recursive={recursive}, prefer_fallback={prefer_fallback}, model='{active_model}')..."
    )

    with _logfire_span(
        "ingest_directory",
        directory=str(target_dir),
        sub_dir=str(sub_dir) if sub_dir else None,
        max_files=max_files,
        wipe=wipe,
        recursive=recursive,
        prefer_fallback=prefer_fallback,
        service_name=INGESTION_SERVICE_NAME,
    ) as root_span:
        files = scan_directory(target_dir, recursive=recursive) if target_dir.exists() else []
        total_scanned = len(files)

        # Apply max_files limit if set
        if max_files is not None and max_files > 0 and len(files) > max_files:
            logger.info(
                f"Limiting ingestion to {max_files} file(s) out of {total_scanned} discovered in '{target_dir}'."
            )
            files = files[:max_files]

        if not files:
            logger.warning(f"No supported document files found in directory: {target_dir}")
            return DirectoryIngestionSummary(
                directory=str(target_dir),
                total_files_scanned=0,
                successful_files=0,
                failed_files=0,
                total_chunks=0,
                total_tables=0,
                total_images=0,
                total_formulas=0,
                total_vectors=0,
                model_used=active_model,
                duration_seconds=0.0,
            )

        workflow = LangChainIngestionWorkflow(
            mysql_manager=db_mgr,
            qdrant_manager=qd_mgr,
            prefer_fallback=prefer_fallback,
            collection_name=active_col,
        )

        summaries: List[FileIngestionSummary] = []
        errors: Dict[str, str] = {}
        total_chunks = 0
        total_tables = 0
        total_images = 0
        total_formulas = 0
        total_vectors = 0

        for idx, file_p in enumerate(files, 1):
            logger.info(f"[{idx}/{len(files)}] Processing '{file_p.name}'...")
            try:
                summary = workflow.invoke(file_p)
                summaries.append(summary)
                total_chunks += summary.chunks_count
                total_tables += summary.tables_count
                total_images += summary.images_count
                total_formulas += summary.formulas_count
                total_vectors += summary.vectors_count
            except Exception as e:
                err_msg = str(e)
                logger.error(f"Failed to ingest '{file_p}': {err_msg}")
                errors[str(file_p)] = err_msg
                if not ignore_errors:
                    raise

        duration = round(time.time() - start_time, 2)
        total_tokens = sum(s.total_tokens for s in summaries)
        total_embedded_tokens = sum(s.total_embedded_tokens for s in summaries)
        avg_tokens_per_chunk = round(total_tokens / total_chunks, 2) if total_chunks > 0 else 0.0
        avg_tokens_per_file = round(total_tokens / len(summaries), 2) if summaries else 0.0
        throughput_chunks = round(total_chunks / duration, 2) if duration > 0 else 0.0
        throughput_tokens = round(total_tokens / duration, 2) if duration > 0 else 0.0

        root_span.set_attributes({
            "ingestion.total_files_scanned": total_scanned,
            "ingestion.successful_files": len(summaries),
            "ingestion.failed_files": len(errors),
            "ingestion.total_chunks": total_chunks,
            "ingestion.total_tokens_consumed": total_tokens,
            "ingestion.total_embedded_tokens": total_embedded_tokens,
            "ingestion.avg_tokens_per_chunk": avg_tokens_per_chunk,
            "ingestion.avg_tokens_per_file": avg_tokens_per_file,
            "ingestion.total_tables": total_tables,
            "ingestion.total_images": total_images,
            "ingestion.total_formulas": total_formulas,
            "ingestion.total_vectors": total_vectors,
            "ingestion.model_used": active_model,
            "ingestion.qdrant_collection": active_col,
            "ingestion.duration_seconds": duration,
            "ingestion.throughput_chunks_per_sec": throughput_chunks,
            "ingestion.throughput_tokens_per_sec": throughput_tokens,
        })

        logger.info(
            f"[Logfire Ingestion Evaluation Report]\n"
            f"====================================================================\n"
            f"  Target Directory:        {target_dir}\n"
            f"  Subdirectory Target:     {sub_dir or 'All'}\n"
            f"  Files Processed:         {len(summaries)} succeeded, {len(errors)} failed (scanned {total_scanned})\n"
            f"  Total Chunks Created:    {total_chunks}\n"
            f"  Total Tokens Consumed:   {total_tokens} (avg {avg_tokens_per_chunk:.1f} tok/chunk, {avg_tokens_per_file:.1f} tok/file)\n"
            f"  Total Embedded Tokens:   {total_embedded_tokens}\n"
            f"  Structured Data (MySQL): {total_tables} tables, {total_images} images, {total_formulas} formulas\n"
            f"  Vectors Indexed (Qdrant):{total_vectors} vectors in collection '{active_col}' ({active_model})\n"
            f"  Total Duration:          {duration}s\n"
            f"  Throughput:              {throughput_chunks:.2f} chunks/s, {throughput_tokens:.1f} tokens/s\n"
            f"===================================================================="
        )

        return DirectoryIngestionSummary(
            directory=str(target_dir),
            total_files_scanned=len(files),
            successful_files=len(summaries),
            failed_files=len(errors),
            total_chunks=total_chunks,
            total_tables=total_tables,
            total_images=total_images,
            total_formulas=total_formulas,
            total_vectors=total_vectors,
            model_used=active_model,
            duration_seconds=duration,
            total_tokens=total_tokens,
            avg_tokens_per_chunk=avg_tokens_per_chunk,
            avg_tokens_per_file=avg_tokens_per_file,
            total_embedded_tokens=total_embedded_tokens,
            throughput_chunks_per_sec=throughput_chunks,
            throughput_tokens_per_sec=throughput_tokens,
            file_summaries=summaries,
            errors=errors,
        )


# ==============================================================================
# QUERY-TIME RETRIEVAL: SUMMARY MATCHING & EXACT MYSQL CONTENT FETCHING
# ==============================================================================


def retrieve_similar_chunks(
    query: str,
    top_k: int = 5,
    prefer_fallback: bool = False,
    collection_name: Optional[str] = None,
    qdrant_manager: Optional[QdrantVectorManager] = None,
    mysql_manager: Optional[MySQLMetadataManager] = None,
) -> List[Dict[str, Any]]:
    """Retrieve semantically relevant items from Qdrant and fetch exact, intact data from MySQL.

    What:
        1. Embeds the user search query using embed_query() with task_type='RETRIEVAL_QUERY'.
           Supports the fallout model when prefer_fallback=True or settings.USE_FALLOUT_MODEL=True.
        2. Executes Cosine vector search against Qdrant collection, matching query against:
           - Embedded table summaries
           - Embedded formula summaries (LaTeX)
           - Embedded image summaries
           - Text chunk contextualized passages
        3. For each retrieved hit:
           - If a table summary matched: grabs `table_id` from metadata, queries MySQL,
             and attaches the complete, intact, uncorrupted table (clean Markdown and JSON data rows).
           - If a formula summary matched: grabs `formula_id`, queries MySQL, and attaches
             the exact LaTeX formula.
           - If an image summary matched: grabs `image_id`, queries MySQL, and attaches
             the image path and base64 URI.
           - If a text chunk matched: formats the contextualized text and joins any tables on that page.
        4. Injects an LLM-ready context block (`formatted_context_for_llm`) ready for prompt injection.

    Args:
        query: User search query.
        top_k: Number of nearest neighbor items to retrieve (default: 5).
        prefer_fallback: Whether to use the fallout model for query embedding.
        collection_name: Qdrant collection name override.
        qdrant_manager: Optional shared Qdrant manager.
        mysql_manager: Optional shared MySQL manager.

    Returns:
        List of enriched result dictionaries containing exact intact tables, LaTeX formulas,
        text passages, and formatted LLM context blocks.
    """
    active_fallback = prefer_fallback or getattr(settings, "USE_FALLOUT_MODEL", False)
    active_model = DEFAULT_FALLBACK_MODEL if active_fallback else DEFAULT_PRIMARY_MODEL

    with _logfire_span(
        "retrieval.retrieve_similar_chunks",
        query=query,
        top_k=top_k,
        prefer_fallback=active_fallback,
        model=active_model,
    ):
        qd_mgr = qdrant_manager or QdrantVectorManager()
        col = collection_name or qd_mgr.collection_name

        # 1. Generate query embedding (asymmetric query vector)
        query_vec = embed_query(
            query=query,
            model=DEFAULT_PRIMARY_MODEL,
            fallback_model=DEFAULT_FALLBACK_MODEL,
            prefer_fallback=active_fallback,
        )

        # 2. Query Qdrant
        search_res = qd_mgr.client.query_points(
            collection_name=col,
            query=query_vec,
            limit=top_k,
            with_payload=True,
        )

        db_mgr = mysql_manager or MySQLMetadataManager()
        results: List[Dict[str, Any]] = []

        # 3. Augment search hits by fetching exact uncorrupted data from MySQL
        for hit in search_res.points:
            payload = hit.payload or {}
            content_type = payload.get("content_type", "text")
            source_file = payload.get("source_file", "")
            page_numbers = payload.get("page_numbers", [])
            table_id = payload.get("table_id")
            formula_id = payload.get("formula_id")
            image_id = payload.get("image_id")

            result_item: Dict[str, Any] = {
                "hit_id": hit.id,
                "score": hit.score,
                "content_type": content_type,
                "source_file": source_file,
                "page_numbers": page_numbers,
                "model_used": payload.get("model_used"),
            }

            # Case A: Table Summary matched -> fetch exact intact table from MySQL
            if content_type == "table_summary" or table_id:
                target_table_id = table_id or payload.get("chunk_id")
                table_record = db_mgr.get_table_by_id(target_table_id) if target_table_id else None

                result_item["table_id"] = target_table_id
                result_item["summary"] = payload.get("summary")
                result_item["caption"] = payload.get("caption")

                if table_record:
                    result_item["intact_table_markdown"] = table_record["markdown"]
                    result_item["intact_table_data_rows"] = table_record["data_rows"]
                    result_item["intact_table_html"] = table_record["html"]
                    result_item["num_rows"] = table_record["num_rows"]
                    result_item["num_cols"] = table_record["num_cols"]
                    caption_str = f" ({table_record['caption']})" if table_record.get("caption") else ""
                    result_item["formatted_context_for_llm"] = (
                        f"### [INTACT TABLE] {target_table_id}{caption_str}\n"
                        f"{table_record['markdown']}\n"
                    )
                else:
                    result_item["formatted_context_for_llm"] = (
                        f"### [TABLE SUMMARY] {target_table_id}\n{payload.get('summary', '')}\n"
                    )

            # Case B: LaTeX Formula Summary matched -> fetch exact formula from MySQL
            elif content_type == "formula_summary" or formula_id:
                target_formula_id = formula_id or payload.get("chunk_id")
                formula_record = db_mgr.get_formula_by_id(target_formula_id) if target_formula_id else None

                result_item["formula_id"] = target_formula_id
                result_item["summary"] = payload.get("summary")

                if formula_record:
                    result_item["latex"] = formula_record["latex"]
                    result_item["context"] = formula_record.get("context")
                    ctx_str = f" (Context: {formula_record['context']})" if formula_record.get("context") else ""
                    result_item["formatted_context_for_llm"] = (
                        f"### [LATEX FORMULA] {target_formula_id}{ctx_str}\n"
                        f"{formula_record['latex']}\n"
                    )
                else:
                    result_item["latex"] = payload.get("latex", "")
                    result_item["formatted_context_for_llm"] = (
                        f"### [LATEX FORMULA] {target_formula_id}\n{payload.get('latex', '')}\n"
                    )

            # Case C: Image Summary matched -> fetch image metadata from MySQL
            elif content_type == "image_summary" or image_id:
                target_image_id = image_id or payload.get("chunk_id")
                image_record = db_mgr.get_image_by_id(target_image_id) if target_image_id else None

                result_item["image_id"] = target_image_id
                result_item["summary"] = payload.get("summary")
                result_item["caption"] = payload.get("caption")

                if image_record:
                    result_item["label"] = image_record["label"]
                    result_item["saved_path"] = image_record["saved_path"]
                    result_item["base64_uri"] = image_record["base64_uri"]
                    result_item["description"] = image_record.get("description")
                    vis_desc = image_record.get("description") or image_record.get("summary", "")
                    result_item["formatted_context_for_llm"] = (
                        f"### [IMAGE ELEMENT] {target_image_id} ({image_record['label']})\n"
                        f"Caption: {image_record.get('caption', 'None')}\n"
                        f"Visual Description: {vis_desc}\n"
                        f"Summary: {image_record.get('summary', '')}\n"
                    )
                else:
                    vis_desc = payload.get("description") or payload.get("summary", "")
                    result_item["formatted_context_for_llm"] = (
                        f"### [IMAGE SUMMARY] {target_image_id}\n"
                        f"Visual Description: {vis_desc}\n"
                    )

            # Case D: Regular Text Chunk matched
            else:
                chunk_id = payload.get("chunk_id", str(hit.id))
                contextualized_text = payload.get("contextualized_text") or payload.get("text", "")
                result_item["chunk_id"] = chunk_id
                result_item["text"] = payload.get("text", "")
                result_item["contextualized_text"] = contextualized_text
                result_item["headings"] = payload.get("headings", [])
                result_item["captions"] = payload.get("captions", [])

                # Check if there are related tables on the same page and attach them
                related_tables = []
                if source_file and page_numbers:
                    doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(Path(source_file).resolve())))
                    with db_mgr.get_session() as session:
                        tbl_records = (
                            session.query(ExtractedTableRecord)
                            .filter(
                                ExtractedTableRecord.document_id == doc_id,
                                ExtractedTableRecord.page_no.in_(page_numbers),
                            )
                            .all()
                        )
                        related_tables = [
                            {
                                "table_id": t.table_id,
                                "caption": t.caption,
                                "markdown": t.markdown,
                            }
                            for t in tbl_records
                        ]

                result_item["related_tables"] = related_tables
                result_item["formatted_context_for_llm"] = (
                    f"### [PASSAGE] {chunk_id}\n{contextualized_text}\n"
                )

            results.append(result_item)

        return results


def build_augmented_llm_context(
    query: str,
    top_k: int = 5,
    prefer_fallback: bool = False,
    collection_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieve top semantic matches and construct an uncorrupted, ready-to-prompt LLM context block.

    Args:
        query: User search query.
        top_k: Number of nearest neighbor items to retrieve.
        prefer_fallback: Whether to prioritize the fallout model.
        collection_name: Qdrant collection override.

    Returns:
        Dictionary with 'llm_context_string', 'hits', 'matched_table_ids', and 'matched_formula_ids'.
    """
    hits = retrieve_similar_chunks(
        query=query,
        top_k=top_k,
        prefer_fallback=prefer_fallback,
        collection_name=collection_name,
    )

    context_blocks: List[str] = []
    matched_table_ids: List[str] = []
    matched_formula_ids: List[str] = []
    matched_image_ids: List[str] = []

    for hit in hits:
        formatted_block = hit.get("formatted_context_for_llm")
        if formatted_block:
            context_blocks.append(formatted_block)

        if hit.get("table_id"):
            matched_table_ids.append(hit["table_id"])
        if hit.get("formula_id"):
            matched_formula_ids.append(hit["formula_id"])
        if hit.get("image_id"):
            matched_image_ids.append(hit["image_id"])

    llm_context_string = "\n---\n".join(context_blocks)
    return {
        "llm_context_string": llm_context_string,
        "hits": hits,
        "matched_table_ids": matched_table_ids,
        "matched_formula_ids": matched_formula_ids,
        "matched_image_ids": matched_image_ids,
    }


# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================


def main() -> None:
    """CLI entry point for running ingestion from terminal."""
    parser = argparse.ArgumentParser(
        description="Enterprise RAG Ingestion Engine (Summary-Indexed Qdrant + MySQL Lookup)"
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        default=getattr(settings, "DATA_DIR", "DATA"),
        help="Directory containing documents to ingest recursively (default: DATA)",
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default=None,
        help="Single file to ingest instead of directory",
    )
    parser.add_argument(
        "--collection",
        "-c",
        type=str,
        default=getattr(settings, "QDRANT_COLLECTION", "dyxi_stag"),
        help="Target Qdrant collection name (default: dyxi_stag)",
    )
    parser.add_argument(
        "--wipe",
        "-w",
        action="store_true",
        help="Wipe all existing data in Qdrant collection and MySQL (wipes only if no file/data-dir is explicitly specified, or wipes before ingesting)",
    )
    parser.add_argument(
        "--wipe-only",
        action="store_true",
        help="Wipe all existing data in Qdrant collection and MySQL without performing any document ingestion",
    )
    parser.add_argument(
        "--sub-dir",
        "--subdir",
        "-s",
        type=str,
        default=None,
        help="Target a specific subdirectory inside the data directory (e.g. 'finetunes', 'noisy')",
    )
    parser.add_argument(
        "--max-files",
        "-m",
        type=int,
        default=None,
        help="Maximum number of files to process",
    )
    parser.add_argument(
        "--fallout",
        action="store_true",
        help="Use fallout/fallback model ('jina-embeddings-v4') instead of primary ('gemini-embedding-2')",
    )
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Disable recursive scanning of child directories",
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        default=None,
        help="Run test retrieval query after ingestion",
    )

    args = parser.parse_args()

    # Determine if this invocation is a dedicated wipe-only operation
    explicit_data_dir = any(arg in sys.argv for arg in ("--data-dir", "-d"))
    wipe_only = args.wipe_only or (args.wipe and not args.file and not explicit_data_dir)

    if wipe_only:
        wipe_result = wipe_all_storage(collection_name=args.collection)
        print("\n--- STORAGE WIPE SUMMARY ---")
        print(json.dumps(wipe_result, indent=2))
        if not args.query:
            return
    elif args.file:
        if args.wipe:
            wipe_all_storage(collection_name=args.collection)
        res = ingest_file(
            file_path=args.file,
            prefer_fallback=args.fallout,
            collection_name=args.collection,
        )
        print("\n--- FILE INGESTION SUMMARY ---")
        print(json.dumps(res.model_dump(), indent=2))
    else:
        summary = ingest_directory(
            directory_path=args.data_dir,
            sub_dir=args.sub_dir,
            max_files=args.max_files,
            wipe=args.wipe,
            recursive=not args.non_recursive,
            prefer_fallback=args.fallout,
            collection_name=args.collection,
        )
        print("\n--- DIRECTORY INGESTION SUMMARY ---")
        print(json.dumps(summary.model_dump(), indent=2))

    if args.query:
        print(f"\n--- TESTING RETRIEVAL FOR QUERY: '{args.query}' ---")
        augmented = build_augmented_llm_context(
            query=args.query,
            top_k=3,
            prefer_fallback=args.fallout,
            collection_name=args.collection,
        )
        print("\n--- GENERATED LLM CONTEXT ---")
        print(augmented["llm_context_string"])


if __name__ == "__main__":
    main()

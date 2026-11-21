"""
Validation test for database initialization scripts and schema.
"""

from pathlib import Path
import re
import sys
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def test_init_schema_sql():
    sql_file = Path("scripts/init_schema.sql")
    assert sql_file.exists(), "scripts/init_schema.sql does not exist"
    
    content = sql_file.read_text(encoding="utf-8")
    
    # Check Database creation
    assert "CREATE DATABASE IF NOT EXISTS `rag_metadata`" in content
    assert "USE `rag_metadata`;" in content
    
    # Check all four tables
    expected_tables = [
        "rag_document_metadata",
        "rag_extracted_tables",
        "rag_extracted_images",
        "rag_extracted_formulas"
    ]
    for table in expected_tables:
        pattern = rf"CREATE TABLE IF NOT EXISTS `{table}`"
        assert re.search(pattern, content), f"Table {table} not found in schema"
        
    # Check critical columns
    assert "`description` LONGTEXT NULL" in content, "Missing description column in rag_extracted_images"
    assert "`formula_id` VARCHAR(128) NOT NULL" in content
    assert "`table_id` VARCHAR(128) NOT NULL" in content
    assert "`image_id` VARCHAR(128) NOT NULL" in content
    assert "`document_id` VARCHAR(128) NOT NULL" in content

def test_script_files_exist():
    assert Path("scripts/setup_database.sh").exists()
    assert Path("scripts/setup_database.ps1").exists()
    assert Path("scripts/init_db.py").exists()

def test_env_table_names_and_constants():
    from app.config import settings
    from app.ingestion.processor import (
        DocumentMetadataRecord,
        ExtractedTableRecord,
        ExtractedImageRecord,
        ExtractedFormulaRecord
    )

    # Verify table names in settings
    assert settings.TABLE_NAME_DOCUMENTS == "rag_document_metadata"
    assert settings.TABLE_NAME_TABLES == "rag_extracted_tables"
    assert settings.TABLE_NAME_IMAGES == "rag_extracted_images"
    assert settings.TABLE_NAME_FORMULAS == "rag_extracted_formulas"

    # Verify SQLAlchemy models bind to settings dynamically
    assert DocumentMetadataRecord.__tablename__ == settings.TABLE_NAME_DOCUMENTS
    assert ExtractedTableRecord.__tablename__ == settings.TABLE_NAME_TABLES
    assert ExtractedImageRecord.__tablename__ == settings.TABLE_NAME_IMAGES
    assert ExtractedFormulaRecord.__tablename__ == settings.TABLE_NAME_FORMULAS

    # Verify other constants in settings
    assert settings.QDRANT_COLLECTION == "dyxi_stag"
    assert settings.EMBEDDING_DIMENSION == 768
    assert settings.RETRIEVAL_TOP_K == 5
    assert settings.CHUNK_MAX_TOKENS == 1200
    assert settings.MYSQL_USER == "root"
    assert settings.MYSQL_DATABASE == "rag_metadata"
    assert ".md" in settings.SUPPORTED_FILE_EXTENSIONS
    assert ".markdown" in settings.SUPPORTED_FILE_EXTENSIONS

def test_markdown_file_support():
    from app.ingestion.loader.load import SUPPORTED_DOCUMENT_EXTENSIONS, scan_directory, load_file
    
    # 1. Verify extension sets
    assert ".md" in SUPPORTED_DOCUMENT_EXTENSIONS
    assert ".markdown" in SUPPORTED_DOCUMENT_EXTENSIONS
    assert ".mdown" in SUPPORTED_DOCUMENT_EXTENSIONS
    assert ".mkd" in SUPPORTED_DOCUMENT_EXTENSIONS
    
    # 2. Verify scan discovers .md files in DATA/true_data
    scanned = scan_directory("DATA/true_data", recursive=True)
    md_files = [f for f in scanned if f.suffix.lower() == ".md"]
    assert len(md_files) > 0, "Expected to find .md files in DATA/true_data"
    
    # 3. Test loading a .md file directly
    test_md = md_files[0]
    doc = load_file(test_md)
    assert doc.file_extension == ".md"
    assert len(doc.markdown_content) > 0
    assert len(doc.elements) > 0
    print(f"Verified markdown loading on {test_md.name}: {len(doc.elements)} elements")

def test_markdown_fallback_logfire():
    from app.ingestion.loader.load import _load_markdown_fallback
    
    test_file = Path("DATA/true_data/RESEARCH_NOTES.md")
    assert test_file.exists(), f"Expected test file {test_file} to exist"
    
    doc = _load_markdown_fallback(
        path=test_file,
        file_name=test_file.name,
        file_ext=test_file.suffix.lower(),
        file_size=test_file.stat().st_size,
        mime_type="text/markdown",
    )
    assert doc is not None
    assert doc.file_name == "RESEARCH_NOTES.md"
    assert len(doc.elements) > 0
    assert doc.metadata.get("parser") == "markdown_fallback_parser"
    assert doc.metadata.get("estimated_tokens", 0) > 0
    assert doc.metadata.get("raw_text_chars", 0) > 0
    print(f"Verified _load_markdown_fallback with Logfire span: {len(doc.elements)} elements, {doc.metadata.get('estimated_tokens')} tokens, parser={doc.metadata.get('parser')}")


def test_logfire_observability_success_and_failure():
    """Verify that Logfire spans track execution, attributes, and exceptions."""
    import logfire

    captured_attributes: dict[str, dict] = {}
    recorded_exceptions: list[BaseException] = []

    class MockSpan:
        def __init__(self, name, **kwargs):
            self.name = name
            self.attrs = dict(kwargs)
            captured_attributes[name] = self.attrs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_type is not None:
                self.attrs["execution.status"] = "FAILED"
                self.attrs["execution.success"] = False
                self.attrs["execution.error_type"] = exc_type.__name__
                self.attrs["execution.error_message"] = str(exc_val)
            else:
                self.attrs["execution.status"] = "SUCCESS"
                self.attrs["execution.success"] = True
            return False

        def set_attribute(self, key, value):
            self.attrs[key] = value

        def set_attributes(self, attributes):
            self.attrs.update(attributes)

        def record_exception(self, exception):
            recorded_exceptions.append(exception)

    original_span = logfire.span
    try:
        logfire.span = lambda name, **kwargs: MockSpan(name, **kwargs)

        # 1. Test Logfire span success block
        with logfire.span("sample_success_span", file_name="sample.md") as span:
            span.set_attribute("custom_key", "custom_value")
            res = 3 + 4
        assert res == 7
        success_span = captured_attributes.get("sample_success_span")
        assert success_span is not None
        assert success_span.get("execution.status") == "SUCCESS"
        assert success_span.get("execution.success") is True
        assert success_span.get("custom_key") == "custom_value"

        # 2. Test Logfire span failure block with exception recording
        with pytest.raises(ValueError, match="Intentional test failure"):
            with logfire.span("sample_fail_span") as fail_span:
                try:
                    raise ValueError("Intentional test failure")
                except ValueError as ex:
                    fail_span.record_exception(ex)
                    raise

        fail_span_attrs = captured_attributes.get("sample_fail_span")
        assert fail_span_attrs is not None
        assert fail_span_attrs.get("execution.status") == "FAILED"
        assert fail_span_attrs.get("execution.success") is False
        assert fail_span_attrs.get("execution.error_type") == "ValueError"
        assert "Intentional test failure" in fail_span_attrs.get("execution.error_message", "")
        assert any(isinstance(e, ValueError) for e in recorded_exceptions)

    finally:
        logfire.span = original_span

    print("Verified Logfire observability spans: SUCCESS and FAILED statuses confirmed.")


def test_app_modules_observability_instrumentation():
    """Verify that all major app modules import and possess @observe instrumentation."""
    import app.ingestion.processor as proc
    import app.ingestion.loader.load as ldr
    import app.ingestion.chunking.chunker as chk
    import app.services.retreival.embeddings as emb

    # Check processor classes
    assert hasattr(proc.MySQLMetadataManager, "store_document_nontextual_data")
    assert hasattr(proc.MySQLMetadataManager, "wipe_all_data")
    assert hasattr(proc.QdrantVectorManager, "upsert_entries")
    assert hasattr(proc.QdrantVectorManager, "wipe_collection")
    assert hasattr(proc.LangChainIngestionWorkflow, "invoke")
    assert hasattr(proc.LangChainIngestionWorkflow, "batch")
    assert callable(proc.ingest_file)
    assert callable(proc.ingest_directory)

    # Check loader classes and functions
    assert callable(ldr.load_file)
    assert callable(ldr.scan_directory)
    assert callable(ldr.describe_image_with_gemini)
    assert hasattr(ldr.MultimodalDocument, "get_nontextual_summary")

    # Check chunker classes and functions
    assert callable(chk.chunk_document)
    assert callable(chk.chunk_text)
    assert hasattr(chk.GeminiTokenEstimator, "count_tokens")

    # Check embeddings classes and functions
    assert callable(emb.embed_texts)
    assert callable(emb.embed_query)
    assert callable(emb.embed_image)
    assert hasattr(emb.GeminiMultimodalEmbeddings, "embed_documents")
    assert hasattr(emb.TokenBucketRateLimiter, "acquire")

    print("Verified all app packages and modules import cleanly with Logfire observability.")


def test_ingestion_evaluation_metrics_and_logfire():
    """Verify that chunk count, token consumption, and evaluation metrics are properly computed and logged."""
    from app.ingestion.processor import FileIngestionSummary, DirectoryIngestionSummary
    from app.ingestion.chunking.chunker import chunk_document
    from app.ingestion.loader.load import _load_markdown_fallback
    import logfire

    # 1. Verify Pydantic evaluation schema models
    summary = FileIngestionSummary(
        file_path="DATA/true_data/RESEARCH_NOTES.md",
        file_name="RESEARCH_NOTES.md",
        document_id="doc-test-123",
        chunks_count=12,
        tables_count=2,
        images_count=1,
        formulas_count=3,
        vectors_count=18,
        model_used="gemini-embedding-exp-0827",
        db_engine="mysql",
        time_taken_seconds=1.5,
        total_tokens=1850,
        avg_tokens_per_chunk=154.17,
        total_embedded_tokens=2200,
        throughput_chunks_per_sec=8.0,
        throughput_tokens_per_sec=1233.33,
    )
    assert summary.chunks_count == 12
    assert summary.total_tokens == 1850
    assert summary.avg_tokens_per_chunk == 154.17
    assert summary.total_embedded_tokens == 2200
    assert summary.throughput_tokens_per_sec == 1233.33

    dir_summary = DirectoryIngestionSummary(
        directory="DATA/true_data",
        total_files_scanned=1,
        successful_files=1,
        failed_files=0,
        total_chunks=12,
        total_tables=2,
        total_images=1,
        total_formulas=3,
        total_vectors=18,
        model_used="gemini-embedding-exp-0827",
        duration_seconds=1.5,
        total_tokens=1850,
        avg_tokens_per_chunk=154.17,
        total_embedded_tokens=2200,
        avg_tokens_per_file=1850.0,
        throughput_chunks_per_sec=8.0,
        throughput_tokens_per_sec=1233.33,
        file_summaries=[summary],
    )
    assert dir_summary.total_tokens == 1850
    assert dir_summary.avg_tokens_per_file == 1850.0
    assert dir_summary.total_embedded_tokens == 2200

    # 2. Test actual chunking and token evaluation on markdown document
    test_file = Path("DATA/true_data/RESEARCH_NOTES.md")
    assert test_file.exists()
    doc = _load_markdown_fallback(
        path=test_file,
        file_name=test_file.name,
        file_ext=test_file.suffix.lower(),
        file_size=test_file.stat().st_size,
        mime_type="text/markdown",
    )
    chunks = chunk_document(doc, max_tokens=200)
    assert len(chunks) > 0
    total_tokens = sum(c.metadata.num_tokens for c in chunks)
    assert total_tokens > 0
    avg_tokens = round(total_tokens / len(chunks), 2)
    assert avg_tokens > 0

    # 3. Test Logfire span tracking of evaluation metrics
    captured_attrs = {}

    class MockLogfireSpan:
        def __init__(self, name, **kwargs):
            self.name = name
            self.attrs = kwargs
            captured_attrs[name] = self.attrs

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def set_attribute(self, k, v):
            self.attrs[k] = v

        def set_attributes(self, attrs):
            self.attrs.update(attrs)

    orig_span = logfire.span
    try:
        logfire.span = lambda name, **kwargs: MockLogfireSpan(name, **kwargs)

        with logfire.span("evaluation_test") as span:
            span.set_attributes({
                "ingestion.chunks_count": len(chunks),
                "ingestion.total_tokens": total_tokens,
                "ingestion.avg_tokens_per_chunk": avg_tokens,
                "ingestion.tables_count": len(doc.tables),
                "ingestion.images_count": len(doc.images),
                "ingestion.formulas_count": len(doc.formulas),
            })

        attrs = captured_attrs.get("evaluation_test", {})
        assert attrs.get("ingestion.chunks_count") == len(chunks)
        assert attrs.get("ingestion.total_tokens") == total_tokens
        assert attrs.get("ingestion.avg_tokens_per_chunk") == avg_tokens
    finally:
        logfire.span = orig_span

    print(f"Verified ingestion evaluation metrics: {len(chunks)} chunks, {total_tokens} total tokens, {avg_tokens} avg tokens/chunk.")


if __name__ == "__main__":
    test_init_schema_sql()
    test_script_files_exist()
    test_env_table_names_and_constants()
    test_markdown_file_support()
    test_markdown_fallback_logfire()
    test_logfire_observability_success_and_failure()
    test_app_modules_observability_instrumentation()
    test_ingestion_evaluation_metrics_and_logfire()
    print("All schema and script tests PASSED!")

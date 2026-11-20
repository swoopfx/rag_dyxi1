"""Ingestion package: multimodal loading, structure-aware chunking, and database ingestion."""

from typing import Any

__all__ = [
    "ingest_file",
    "ingest_directory",
    "retrieve_similar_chunks",
    "wipe_all_storage",
    "LangChainIngestionWorkflow",
    "MySQLMetadataManager",
    "QdrantVectorManager",
    "FileIngestionSummary",
    "DirectoryIngestionSummary",
    "DocumentMetadataRecord",
    "ExtractedTableRecord",
    "ExtractedImageRecord",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        import importlib
        module = importlib.import_module("app.ingestion.processor")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

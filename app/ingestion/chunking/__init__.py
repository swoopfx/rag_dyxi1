"""Document chunking package using Docling HybridChunker tailored for Gemini Embedding 2."""

from app.ingestion.chunking.chunker import (
    ChunkMetadata,
    GeminiTokenEstimator,
    ProcessedChunk,
    build_hybrid_chunker,
    build_threaded_pdf_options,
    chunk_docling_document,
    chunk_document,
    is_digital_pdf,
    slice_pdf_to_page_batches,
    stream_chunks,
    stream_convert_and_chunk,
)

__all__ = [
    "ChunkMetadata",
    "GeminiTokenEstimator",
    "ProcessedChunk",
    "build_hybrid_chunker",
    "build_threaded_pdf_options",
    "chunk_docling_document",
    "chunk_document",
    "is_digital_pdf",
    "slice_pdf_to_page_batches",
    "stream_chunks",
    "stream_convert_and_chunk",
]

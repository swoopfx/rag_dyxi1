"""Document loading package for multimodal ingestion."""

from app.ingestion.loader.load import (
    SUPPORTED_DOCUMENT_EXTENSIONS,
    BoundingBox,
    DocumentElement,
    ExtractedFormula,
    ExtractedImage,
    ExtractedTable,
    MultimodalDocument,
    build_docling_converter,
    describe_image_with_gemini,
    image_to_base64,
    load_directory,
    load_document,
    load_file,
    load_multimodal_document,
    scan_directory,
)

__all__ = [
    "load_file",
    "load_document",
    "load_multimodal_document",
    "load_directory",
    "scan_directory",
    "SUPPORTED_DOCUMENT_EXTENSIONS",
    "MultimodalDocument",
    "ExtractedImage",
    "ExtractedTable",
    "ExtractedFormula",
    "DocumentElement",
    "BoundingBox",
    "build_docling_converter",
    "describe_image_with_gemini",
    "image_to_base64",
]

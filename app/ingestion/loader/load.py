"""Multimodal document loader using Docling.

This module provides comprehensive document loading and multimodal content
extraction (text, headings, lists, tables, formulas, code, and images/figures)
with support for PDFs, images, DOCX, PPTX, HTML, and other file formats.
"""

from __future__ import annotations

import base64
import contextlib
import io
import mimetypes
from pathlib import Path
import re
from typing import Any, Dict, Iterator, List, Literal, Optional, Sequence, Union

import pandas as pd
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

# Google GenAI imports for Vision
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None  # type: ignore
    types = None  # type: ignore

from app.config import settings

# Docling imports
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption
from docling_core.types.doc import (
    CodeItem,
    DoclingDocument,
    FormulaItem,
    ListItem,
    PictureItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
)

# Logfire Observability Setup
try:
    import logfire

    _logfire_token = getattr(settings, "LOGFIRE_TOKEN", None) or os.getenv("LOGFIRE_TOKEN")
    logfire.configure(
        send_to_logfire="if-token-present",
        token=_logfire_token,
        service_name="rag-loader-service",
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


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg",
                    ".tiff", ".tif", ".bmp", ".webp", ".gif"}
SUPPORTED_DOCUMENT_EXTENSIONS: set[str] = {
    ".pdf",
    ".docx",
    ".doc",
    ".pptx",
    ".ppt",
    ".xlsx",
    ".xls",
    ".csv",
    ".html",
    ".htm",
    ".md",
    ".markdown",
    ".mdown",
    ".mkd",
    ".txt",
    ".text",
    ".json",
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".tif",
    ".bmp",
    ".webp",
    ".gif",
}


# ==============================================================================
# DATA MODELS FOR MULTIMODAL CONTENT
# ==============================================================================


class BoundingBox(BaseModel):
    """Bounding box coordinates for visual grounding of document elements."""

    l: float = Field(..., description="Left coordinate")
    t: float = Field(..., description="Top coordinate")
    r: float = Field(..., description="Right coordinate")
    b: float = Field(..., description="Bottom coordinate")
    coord_origin: str = Field(
        default="BOTTOMLEFT", description="Coordinate origin (e.g. BOTTOMLEFT, TOPLEFT)"
    )

    @property
    def width(self) -> float:
        return abs(self.r - self.l)

    @property
    def height(self) -> float:
        return abs(self.t - self.b)


# ==============================================================================
# GEMINI VISION IMAGE DESCRIPTION (LOW-LEVEL VISION MODEL)
# ==============================================================================


def describe_image_with_gemini(
    image: Union[Image.Image, bytes, bytearray, str, Path],
    model: Optional[str] = None,
    fallback_model: Optional[str] = None,
    prompt: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """Generate a rich, search-optimized visual description using Gemini Flash Vision.

    What:
        Inspects an image (architecture diagram, plot, photo, flowchart, UI mock) and
        generates a dense 2-3 sentence semantic description for vector embedding and
        relational database storage.

    Why:
        Enables high-recall vector search against uncaptioned diagrams, figures, and plots
        by converting visual content and embedded text into search-indexable keywords.
    """
    key = api_key or getattr(settings, "GEMINI_API_KEY",
                             None) or os.getenv("GEMINI_API_KEY")
    if not key or genai is None or types is None:
        logger.debug(
            "Gemini client or API key not available for vision description.")
        return ""

    # Convert input to image bytes and mime type
    img_bytes: bytes = b""
    mime = "image/png"

    try:
        if isinstance(image, Image.Image):
            buf = io.BytesIO()
            fmt = image.format or "PNG"
            save_img = image
            if fmt.upper() == "JPEG" and save_img.mode in ("RGBA", "P"):
                save_img = save_img.convert("RGB")
            save_img.save(buf, format=fmt)
            img_bytes = buf.getvalue()
            mime = f"image/{fmt.lower()}"
        elif isinstance(image, (bytes, bytearray)):
            img_bytes = bytes(image)
        elif isinstance(image, (str, Path)):
            p = Path(image)
            if p.exists() and p.is_file():
                img_bytes = p.read_bytes()
                guessed, _ = mimetypes.guess_type(str(p))
                if guessed:
                    mime = guessed
    except Exception as conv_err:
        logger.debug(
            f"Failed to convert image to bytes for vision model: {conv_err}")
        return ""

    if not img_bytes:
        return ""

    target_model = model or getattr(
        settings, "GEMINI_VISION_MODEL", "gemini-flash-latest")
    fallback = fallback_model or getattr(
        settings, "FALLBACK_VISION_MODEL", "gemini-3.1-flash-lite")
    active_prompt = prompt or (
        "Describe this image concisely in 2-3 sentences for search retrieval and indexing. "
        "Explicitly mention: 1) What it depicts (diagram, chart, UI, flowchart, photo), "
        "2) Key visible labels, headings, and data numbers, and "
        "3) Core findings or architectural connections."
    )

    client = genai.Client(api_key=key)
    part = types.Part.from_bytes(data=img_bytes, mime_type=mime)

    with _logfire_span("gemini_vision.describe_image", model=target_model, byte_size=len(img_bytes)):
        for attempt_model in [target_model, fallback]:
            try:
                res = client.models.generate_content(
                    model=attempt_model,
                    contents=[part, active_prompt],
                )
                if res and res.text:
                    desc = res.text.strip()
                    logger.debug(
                        f"Gemini Flash Vision generated description ({len(desc)} chars) via '{attempt_model}'.")
                    return desc
            except Exception as e:
                logger.warning(
                    f"Gemini vision description attempt with '{attempt_model}' failed: {e}")
                continue

    return ""


class ExtractedImage(BaseModel):
    """Extracted image or visual element (figure, chart, picture, diagram)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: int = Field(..., description="Order index of the extracted image")
    caption: Optional[str] = Field(
        default=None, description="Caption or text accompanying the image")
    label: Optional[str] = Field(
        default="picture", description="Element label (picture, chart, etc.)")
    page_no: Optional[int] = Field(
        default=None, description="Page number where the image was found")
    bbox: Optional[BoundingBox] = Field(
        default=None, description="Bounding box on page")
    width: Optional[int] = Field(default=None, description="Image pixel width")
    height: Optional[int] = Field(
        default=None, description="Image pixel height")
    format: str = Field(
        default="PNG", description="Image format (PNG, JPEG, etc.)")
    image: Optional[Image.Image] = Field(
        default=None, description="PIL Image instance")
    base64_uri: Optional[str] = Field(
        default=None, description="Base64 Data URI string (e.g. data:image/png;base64,...)"
    )
    saved_path: Optional[str] = Field(
        default=None, description="File path if the image was persisted to disk"
    )
    annotations: List[str] = Field(
        default_factory=list, description="Any classifications or annotations"
    )
    description: Optional[str] = Field(
        default=None,
        description="Detailed visual description generated by Gemini Flash Vision",
    )
    summary: Optional[str] = Field(
        default=None, description="Concise semantic summary of the visual element for vector search"
    )

    def generate_summary(self, use_vision: bool = True) -> str:
        """Generate a semantic summary of the visual element for vector search."""
        if self.summary and self.description:
            return self.summary

        # Auto-describe using low-level Gemini Flash Vision if enabled
        enable_vision = getattr(
            settings, "ENABLE_VISION_IMAGE_DESCRIPTION", True)
        if use_vision and enable_vision and not self.description:
            img_source = self.image or self.saved_path
            if img_source is not None:
                try:
                    desc = describe_image_with_gemini(img_source)
                    if desc:
                        self.description = desc
                except Exception as e:
                    logger.debug(f"Gemini vision description skipped: {e}")

        parts = [f"Visual element ({self.label}):"]
        if self.caption:
            parts.append(f"Caption: '{self.caption}'.")
        if self.description:
            parts.append(f"Visual Description: {self.description}")
        if self.page_no:
            parts.append(f"Page {self.page_no}.")
        if self.annotations:
            parts.append(f"Annotations: {', '.join(self.annotations)}.")
        if self.saved_path:
            parts.append(f"File: {Path(self.saved_path).name}.")

        summary_val = " ".join(parts).strip()
        self.summary = summary_val
        return summary_val

    def save(self, output_path: Union[str, Path]) -> Path:
        """Save the image to disk."""
        target_path = Path(output_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if self.image is not None:
            self.image.save(target_path)
            self.saved_path = str(target_path.resolve())
            return target_path
        raise ValueError("Cannot save image: PIL Image instance is None")


class ExtractedTable(BaseModel):
    """Extracted structured table from the document."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: int = Field(..., description="Order index of the table")
    markdown: str = Field(..., description="Table rendered as Markdown")
    html: Optional[str] = Field(
        default=None, description="Table rendered as HTML")
    dataframe: Optional[pd.DataFrame] = Field(
        default=None, description="Pandas DataFrame representation"
    )
    data_rows: List[List[Any]] = Field(
        default_factory=list, description="Raw rows of the table for serialization"
    )
    num_rows: int = Field(default=0, description="Number of table rows")
    num_cols: int = Field(default=0, description="Number of table columns")
    caption: Optional[str] = Field(
        default=None, description="Caption accompanying the table")
    page_no: Optional[int] = Field(
        default=None, description="Page number of the table")
    bbox: Optional[BoundingBox] = Field(
        default=None, description="Bounding box on the page")
    table_image: Optional[ExtractedImage] = Field(
        default=None, description="Visual crop of the table if enabled"
    )
    summary: Optional[str] = Field(
        default=None, description="Concise semantic summary of the table for vector search"
    )

    def generate_summary(self) -> str:
        """Generate a semantic summary of table schema, columns, and sample entries."""
        if self.summary:
            return self.summary

        parts = []
        if self.caption:
            parts.append(f"Table title/caption: {self.caption}.")
        if self.page_no:
            parts.append(f"Located on page {self.page_no}.")

        cols: List[str] = []
        if self.dataframe is not None and not self.dataframe.empty:
            cols = [str(c) for c in self.dataframe.columns]
        elif self.data_rows and len(self.data_rows) > 0:
            cols = [str(c) for c in self.data_rows[0]]

        if cols:
            parts.append(f"Columns: {', '.join(cols)}.")
        parts.append(
            f"Dimensions: {self.num_rows} rows by {self.num_cols} columns.")

        sample_items = []
        if self.data_rows and len(self.data_rows) > 1:
            for r in self.data_rows[1:4]:
                sample_items.append(" | ".join(str(val)
                                    for val in r if str(val).strip()))
        elif self.dataframe is not None and not self.dataframe.empty:
            for _, row in self.dataframe.head(3).iterrows():
                sample_items.append(" | ".join(str(val)
                                    for val in row.values if str(val).strip()))

        if sample_items:
            parts.append(f"Sample data entries: {'; '.join(sample_items)}.")

        summary_val = " ".join(parts).strip(
        ) or "Structured table containing data."
        self.summary = summary_val
        return summary_val


class ExtractedFormula(BaseModel):
    """Extracted mathematical formula or equation formatted in LaTeX."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    index: int = Field(..., description="Order index of the formula")
    latex: str = Field(...,
                       description="Raw LaTeX formula string (e.g. $E=mc^2$ or display math)")
    context: Optional[str] = Field(
        default=None, description="Surrounding contextual text or heading")
    page_no: Optional[int] = Field(
        default=None, description="Page number where formula appears")
    bbox: Optional[BoundingBox] = Field(
        default=None, description="Bounding box on page")
    summary: Optional[str] = Field(
        default=None, description="Semantic summary for vector search embedding"
    )

    def generate_summary(self) -> str:
        """Generate a searchable summary of the LaTeX formula."""
        if self.summary:
            return self.summary
        parts = ["Mathematical Formula in LaTeX:", self.latex]
        if self.context:
            parts.append(f"Context: {self.context}.")
        if self.page_no:
            parts.append(f"Page: {self.page_no}.")
        summary_val = " ".join(parts).strip()
        self.summary = summary_val
        return summary_val


class DocumentElement(BaseModel):
    """An individual element from the document in reading order."""

    index: int = Field(..., description="Reading order index")
    type: Literal["heading", "text", "list_item", "code", "formula", "table", "picture", "other"] = (
        Field(..., description="Type of document element")
    )
    text: str = Field(...,
                      description="Extracted text or content of the element")
    level: Optional[int] = Field(
        default=None, description="Heading level if applicable")
    page_no: Optional[int] = Field(default=None, description="Page number")
    bbox: Optional[BoundingBox] = Field(
        default=None, description="Bounding box")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional item metadata")


class MultimodalDocument(BaseModel):
    """Comprehensive container for a multi-modal document extracted with Docling."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    file_path: Optional[str] = Field(
        default=None, description="Source file path or URL")
    file_name: str = Field(..., description="Source file name")
    file_extension: str = Field(...,
                                description="File extension (e.g. .pdf, .png)")
    file_size_bytes: Optional[int] = Field(
        default=None, description="Source file size in bytes")
    mime_type: Optional[str] = Field(
        default=None, description="Detected MIME type")
    num_pages: int = Field(
        default=1, description="Number of pages in the document")
    markdown_content: str = Field(
        default="", description="Full document content formatted as Markdown"
    )
    plain_text: str = Field(
        default="", description="Full document plain text content")
    elements: List[DocumentElement] = Field(
        default_factory=list, description="Extracted elements in reading order"
    )
    tables: List[ExtractedTable] = Field(
        default_factory=list, description="Extracted structured tables"
    )
    images: List[ExtractedImage] = Field(
        default_factory=list, description="Extracted images, figures, and charts"
    )
    formulas: List[ExtractedFormula] = Field(
        default_factory=list, description="Extracted mathematical formulas formatted in LaTeX"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Document metadata and statistics"
    )
    raw_document: Optional[DoclingDocument] = Field(
        default=None, description="Raw DoclingDocument for downstream low-level operations"
    )

    @property
    def has_images(self) -> bool:
        """Check if document contains extracted images."""
        return len(self.images) > 0

    @property
    def has_tables(self) -> bool:
        """Check if document contains extracted tables."""
        return len(self.tables) > 0

    @property
    def has_formulas(self) -> bool:
        """Check if document contains extracted formulas."""
        return len(self.formulas) > 0

    @property
    def has_nontextual_content(self) -> bool:
        """Check if document contains any extracted tables, images, or formulas."""
        return self.has_images or self.has_tables or self.has_formulas

    def get_nontextual_summary(self) -> Dict[str, Any]:
        """Summary of nontextual tables, images, and formulas for logging and database records."""
        return {
            "tables_count": len(self.tables),
            "images_count": len(self.images),
            "formulas_count": len(self.formulas),
            "table_captions": [t.caption for t in self.tables if t.caption],
            "image_captions": [img.caption for img in self.images if img.caption],
            "image_labels": [img.label for img in self.images if img.label],
            "formulas": [f.latex for f in self.formulas],
        }

    def to_langchain_documents(self) -> List[Any]:
        """Convert document content into standard LangChain Document instances."""
        from langchain_core.documents import Document

        return [
            Document(
                page_content=self.markdown_content or self.plain_text,
                metadata={
                    "source": self.file_path or self.file_name,
                    "file_name": self.file_name,
                    "file_extension": self.file_extension,
                    "num_pages": self.num_pages,
                    "has_tables": self.has_tables,
                    "has_images": self.has_images,
                    "num_tables": len(self.tables),
                    "num_images": len(self.images),
                    **self.metadata,
                },
            )
        ]

    def get_headings(self) -> List[DocumentElement]:
        """Return all heading elements from the document."""
        return [elem for elem in self.elements if elem.type == "heading"]

    def to_dict(self, include_raw: bool = False, include_pil: bool = False) -> Dict[str, Any]:
        """Convert document to a JSON-serializable dictionary.

        Args:
            include_raw: Whether to include the raw DoclingDocument object.
            include_pil: Whether to retain PIL Image references in extracted images.
        """
        data = self.model_dump(
            exclude={"raw_document"} if not include_raw else None
        )
        if not include_pil and "images" in data:
            for img in data["images"]:
                img.pop("image", None)
        if not include_raw and "tables" in data:
            for tbl in data["tables"]:
                tbl.pop("dataframe", None)
        return data


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert a PIL Image to a Base64 Data URI string."""
    buffer = io.BytesIO()
    save_img = image
    fmt = format.upper()
    if fmt == "JPEG" and save_img.mode in ("RGBA", "P"):
        save_img = save_img.convert("RGB")
    save_img.save(buffer, format=fmt)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{encoded}"


def _extract_bbox_from_prov(item: Any) -> tuple[Optional[int], Optional[BoundingBox]]:
    """Extract page number and bounding box coordinates from item provenance."""
    page_no: Optional[int] = None
    bbox: Optional[BoundingBox] = None
    prov_list = getattr(item, "prov", None)
    if prov_list and len(prov_list) > 0:
        p = prov_list[0]
        page_no = getattr(p, "page_no", None)
        b = getattr(p, "bbox", None)
        if b is not None:
            bbox = BoundingBox(
                l=float(b.l),
                t=float(b.t),
                r=float(b.r),
                b=float(b.b),
                coord_origin=str(getattr(b, "coord_origin", "BOTTOMLEFT")),
            )
    return page_no, bbox


def build_docling_converter(
    do_ocr: bool = True,
    do_table_structure: bool = True,
    generate_page_images: bool = True,
    images_scale: float = 2.0,
    do_formula_enrichment: bool = False,
    do_code_enrichment: bool = False,
) -> DocumentConverter:
    """Build and configure a Docling DocumentConverter with multimodal pipeline options."""
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = do_ocr
    pipeline_options.do_table_structure = do_table_structure
    pipeline_options.generate_page_images = generate_page_images
    pipeline_options.images_scale = images_scale
    pipeline_options.do_formula_enrichment = do_formula_enrichment
    pipeline_options.do_code_enrichment = do_code_enrichment

    format_options = {
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options),
    }

    return DocumentConverter(format_options=format_options)


def _load_markdown_fallback(
    path: Path,
    file_name: str,
    file_ext: str,
    file_size: Optional[int],
    mime_type: Optional[str],
) -> MultimodalDocument:
    """Resilient fallback parser for markdown files (.md, .markdown) if converter encounters issues."""
    with _logfire_span(
        "loader.load_markdown_fallback",
        file_name=file_name,
        file_path=str(path),
        file_size_bytes=file_size,
        file_extension=file_ext,
        mime_type=mime_type or "text/markdown",
    ) as span:
        try:
            raw_text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.error(f"Failed to read markdown file {path}: {e}")
            span.record_exception(e)
            raise

        elements: List[DocumentElement] = []
        lines = raw_text.splitlines()
        for idx, line in enumerate(lines):
            line_s = line.strip()
            if not line_s:
                continue
            if line_s.startswith("#"):
                level = len(line_s) - len(line_s.lstrip("#"))
                elements.append(
                    DocumentElement(
                        index=idx,
                        type="heading",
                        text=line_s.lstrip("#").strip(),
                        page_no=1,
                        metadata={"level": level},
                    )
                )
            elif line_s.startswith(("- ", "* ", "+ ")) or (
                len(line_s) > 2 and line_s[:1].isdigit(
                ) and line_s[1:3] in {". ", ") "}
            ):
                elements.append(
                    DocumentElement(
                        index=idx,
                        type="list_item",
                        text=line_s,
                        page_no=1,
                    )
                )
            else:
                elements.append(
                    DocumentElement(
                        index=idx,
                        type="text",
                        text=line_s,
                        page_no=1,
                    )
                )

        raw_chars = len(raw_text)
        est_tokens = max(1, max(int(len(raw_text.split()) * 1.3), raw_chars // 4)) if raw_chars > 0 else 0
        span.set_attributes({
            "num_elements": len(elements),
            "num_lines": len(lines),
            "content_length_chars": raw_chars,
            "raw_text_chars": raw_chars,
            "estimated_tokens": est_tokens,
            "file_extension": file_ext,
            "parser": "markdown_fallback_parser",
        })

        logger.info(
            f"[Logfire] Fallback markdown parsing completed for '{file_name}': "
            f"{len(elements)} elements extracted, ~{est_tokens} estimated tokens."
        )

        return MultimodalDocument(
            file_path=str(path),
            file_name=file_name,
            file_extension=file_ext,
            file_size_bytes=file_size,
            mime_type=mime_type or "text/markdown",
            num_pages=1,
            markdown_content=raw_text,
            plain_text=raw_text,
            elements=elements,
            tables=[],
            images=[],
            formulas=[],
            metadata={
                "num_pages": 1,
                "num_elements": len(elements),
                "num_tables": 0,
                "num_images": 0,
                "file_extension": file_ext,
                "mime_type": mime_type or "text/markdown",
                "raw_text_chars": raw_chars,
                "estimated_tokens": est_tokens,
                "parser": "markdown_fallback_parser",
            },
            raw_document=None,
        )


# ==============================================================================
# MAIN MULTIMODAL LOADER FUNCTION
# ==============================================================================


def load_file(
    file_path: Union[str, Path],
    extract_images: bool = True,
    extract_tables: bool = True,
    include_base64_images: bool = True,
    include_table_images: bool = False,
    save_images_dir: Optional[Union[str, Path]] = None,
    do_ocr: bool = True,
    do_table_structure: bool = True,
    images_scale: float = 2.0,
    converter: Optional[DocumentConverter] = None,
    **kwargs: Any,
) -> MultimodalDocument:
    """Load a document or image file and extract rich multimodal content using Docling.

    Handles PDFs, images (PNG, JPG, TIFF, WEBP, etc.), Office documents (DOCX, PPTX, XLSX),
    HTML, and markdown. Extracts full markdown text, reading-order elements, structured
    tables (Markdown, HTML, DataFrame), and visual figures/pictures (with PIL Image and
    Base64 URIs for multimodal LLM consumption).

    Args:
        file_path: Path to the target file or document URL.
        extract_images: Whether to extract visual pictures/figures from the document.
        extract_tables: Whether to extract structured tables.
        include_base64_images: Whether to generate Base64 data URIs for extracted images.
        include_table_images: Whether to also render visual crop images for tables.
        save_images_dir: Optional directory where extracted images should be saved on disk.
        do_ocr: Whether to run OCR on images and scanned pages.
        do_table_structure: Whether to extract fine-grained table structures.
        images_scale: Scale factor for rendering visual elements (default 2.0 for high fidelity).
        converter: Optional custom Docling DocumentConverter instance.
        **kwargs: Additional parameters passed to converter or pipeline.

    Returns:
        MultimodalDocument: Structured container with text, elements, tables, and images.
    """
    is_url = isinstance(file_path, str) and (
        file_path.startswith("http://") or file_path.startswith("https://")
    )
    path: Optional[Path] = None

    if not is_url:
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Source file does not exist: {path}")
        file_name = path.name
        file_ext = path.suffix.lower()
        file_size = path.stat().st_size
        mime_type, _ = mimetypes.guess_type(path)
        source: Union[Path, str] = path
    else:
        file_name = file_path.split("/")[-1] or "document"
        file_ext = Path(file_name).suffix.lower()
        file_size = None
        mime_type, _ = mimetypes.guess_type(file_name)
        source = file_path

    logger.info(
        f"Loading multimodal document from {source} (type: {file_ext})")

    # Quick fallback for standard plain text files without requiring Docling converter
    if path is not None and file_ext in {".txt", ".text", ".log", ".json", ".yaml", ".yml"}:
        try:
            raw_text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.error(f"Failed to read text file {path}: {e}")
            raise

        elem = DocumentElement(
            index=0,
            type="text",
            text=raw_text,
            page_no=1,
            bbox=None,
        )
        return MultimodalDocument(
            file_path=str(path),
            file_name=file_name,
            file_extension=file_ext,
            file_size_bytes=file_size,
            mime_type=mime_type or "text/plain",
            num_pages=1,
            markdown_content=raw_text,
            plain_text=raw_text,
            elements=[elem],
            tables=[],
            images=[],
            metadata={
                "num_pages": 1,
                "num_elements": 1,
                "num_tables": 0,
                "num_images": 0,
                "file_extension": file_ext,
                "mime_type": mime_type or "text/plain",
                "raw_text_chars": len(raw_text),
                "estimated_tokens": max(1, max(int(len(raw_text.split()) * 1.3), len(raw_text) // 4)) if raw_text else 0,
            },
            raw_document=None,
        )

    # Prepare images destination directory if requested
    target_img_dir: Optional[Path] = None
    if save_images_dir is not None:
        target_img_dir = Path(save_images_dir)
        target_img_dir.mkdir(parents=True, exist_ok=True)

    # Initialize converter with multimodal options if not supplied
    if converter is None:
        generate_page_images = extract_images or include_table_images
        converter = build_docling_converter(
            do_ocr=do_ocr,
            do_table_structure=do_table_structure,
            generate_page_images=generate_page_images,
            images_scale=images_scale,
        )

    # Convert document via Docling
    try:
        conversion_result = converter.convert(source, **kwargs)
        doc: DoclingDocument = conversion_result.document
    except Exception as e:
        logger.error(f"Docling conversion failed for {source}: {e}")
        if path is not None and file_ext in {".md", ".markdown", ".mdown", ".mkd"}:
            logger.info(
                f"Falling back to resilient native markdown parser for {source}")
            return _load_markdown_fallback(path, file_name, file_ext, file_size, mime_type)
        raise RuntimeError(
            f"Failed to extract document content from {source}: {e}") from e

    # Extract overall markdown and text
    markdown_content = doc.export_to_markdown()
    plain_text = doc.export_to_text()

    # --------------------------------------------------------------------------
    # 1. Extract Elements in Reading Order
    # --------------------------------------------------------------------------
    elements: List[DocumentElement] = []
    elem_idx = 0

    for item, _ in doc.iterate_items():
        page_no, bbox = _extract_bbox_from_prov(item)
        elem_type: Literal[
            "heading", "text", "list_item", "code", "formula", "table", "picture", "other"
        ] = "other"
        elem_text = ""
        elem_level: Optional[int] = None
        meta: Dict[str, Any] = {}

        if isinstance(item, SectionHeaderItem):
            elem_type = "heading"
            elem_text = getattr(item, "text", "")
            elem_level = getattr(item, "level", 1)
        elif isinstance(item, ListItem):
            elem_type = "list_item"
            elem_text = getattr(item, "text", "")
        elif isinstance(item, CodeItem):
            elem_type = "code"
            elem_text = getattr(item, "text", "")
        elif isinstance(item, FormulaItem):
            elem_type = "formula"
            elem_text = getattr(item, "text", "")
        elif isinstance(item, TableItem):
            elem_type = "table"
            try:
                elem_text = item.export_to_markdown(doc)
            except Exception:
                elem_text = "[Table]"
        elif isinstance(item, PictureItem):
            elem_type = "picture"
            caption = None
            try:
                caption = item.caption_text(doc)
            except Exception:
                pass
            elem_text = f"![{caption or 'Figure'}]"
            meta["caption"] = caption
        elif isinstance(item, TextItem):
            elem_type = "text"
            elem_text = getattr(item, "text", "")
        else:
            elem_text = getattr(item, "text", "") or ""

        if elem_text.strip() or elem_type in ("table", "picture"):
            elements.append(
                DocumentElement(
                    index=elem_idx,
                    type=elem_type,
                    text=elem_text,
                    level=elem_level,
                    page_no=page_no,
                    bbox=bbox,
                    metadata=meta,
                )
            )
            elem_idx += 1

    # --------------------------------------------------------------------------
    # 2. Extract Structured Tables
    # --------------------------------------------------------------------------
    extracted_tables: List[ExtractedTable] = []
    if extract_tables:
        for idx, table_item in enumerate(doc.tables):
            page_no, bbox = _extract_bbox_from_prov(table_item)

            # Markdown export
            try:
                table_md = table_item.export_to_markdown(doc)
            except Exception as e:
                logger.warning(
                    f"Failed to export table {idx} to markdown: {e}")
                table_md = ""

            # HTML export
            table_html: Optional[str] = None
            try:
                table_html = table_item.export_to_html(doc)
            except Exception:
                pass

            # DataFrame & serializable rows
            df: Optional[pd.DataFrame] = None
            data_rows: List[List[Any]] = []
            num_rows = 0
            num_cols = 0
            try:
                df = table_item.export_to_dataframe(doc)
                if df is not None:
                    num_rows, num_cols = df.shape
                    # Include header + rows for clean serialization
                    data_rows = [list(df.columns)] + df.values.tolist()
            except Exception as e:
                logger.debug(
                    f"DataFrame export for table {idx} failed or not supported: {e}")

            # Caption
            caption: Optional[str] = None
            try:
                caption = table_item.caption_text(doc)
            except Exception:
                pass

            # Visual table crop
            table_img_obj: Optional[ExtractedImage] = None
            if include_table_images:
                try:
                    pil_img = table_item.get_image(doc)
                    if pil_img is not None:
                        b64_uri = (
                            image_to_base64(pil_img, format="PNG")
                            if include_base64_images
                            else None
                        )
                        saved_path = None
                        if target_img_dir is not None:
                            img_file = target_img_dir / f"table_{idx + 1}.png"
                            pil_img.save(img_file)
                            saved_path = str(img_file.resolve())

                        table_img_obj = ExtractedImage(
                            index=idx,
                            caption=caption,
                            label="table",
                            page_no=page_no,
                            bbox=bbox,
                            width=pil_img.width,
                            height=pil_img.height,
                            format="PNG",
                            image=pil_img,
                            base64_uri=b64_uri,
                            saved_path=saved_path,
                        )
                except Exception as e:
                    logger.debug(f"Could not crop image for table {idx}: {e}")

            extracted_tables.append(
                ExtractedTable(
                    index=idx,
                    markdown=table_md,
                    html=table_html,
                    dataframe=df,
                    data_rows=data_rows,
                    num_rows=num_rows,
                    num_cols=num_cols,
                    caption=caption,
                    page_no=page_no,
                    bbox=bbox,
                    table_image=table_img_obj,
                )
            )

    # --------------------------------------------------------------------------
    # 3. Extract Images, Figures, and Charts
    # --------------------------------------------------------------------------
    extracted_images: List[ExtractedImage] = []
    if extract_images:
        # A) Embedded pictures found in document
        for idx, pic_item in enumerate(doc.pictures):
            page_no, bbox = _extract_bbox_from_prov(pic_item)
            caption: Optional[str] = None
            try:
                caption = pic_item.caption_text(doc)
            except Exception:
                pass

            pil_img: Optional[Image.Image] = None
            try:
                pil_img = pic_item.get_image(doc)
            except Exception as e:
                logger.debug(
                    f"Could not extract PIL image for picture {idx}: {e}")

            b64_uri: Optional[str] = None
            saved_path: Optional[str] = None
            width: Optional[int] = None
            height: Optional[int] = None

            if pil_img is not None:
                width = pil_img.width
                height = pil_img.height
                if include_base64_images:
                    try:
                        b64_uri = image_to_base64(pil_img, format="PNG")
                    except Exception as e:
                        logger.debug(
                            f"Base64 conversion failed for picture {idx}: {e}")

                if target_img_dir is not None:
                    img_file = target_img_dir / f"image_{idx + 1}.png"
                    try:
                        pil_img.save(img_file)
                        saved_path = str(img_file.resolve())
                    except Exception as e:
                        logger.warning(
                            f"Failed to save image {idx} to {img_file}: {e}")

            annotations: List[str] = []
            try:
                annots = pic_item.get_annotations()
                if annots:
                    annotations = [str(a) for a in annots]
            except Exception:
                pass

            extracted_images.append(
                ExtractedImage(
                    index=idx,
                    caption=caption,
                    label=getattr(pic_item, "label", "picture") or "picture",
                    page_no=page_no,
                    bbox=bbox,
                    width=width,
                    height=height,
                    format="PNG",
                    image=pil_img,
                    base64_uri=b64_uri,
                    saved_path=saved_path,
                    annotations=annotations,
                )
            )

        # B) If input file itself is an image format and doc.pictures is empty,
        # preserve the input image itself so visual content is never lost.
        if len(extracted_images) == 0 and path is not None and file_ext in IMAGE_EXTENSIONS:
            try:
                with Image.open(path) as raw_img:
                    raw_copy = raw_img.copy()
                    b64_uri = (
                        image_to_base64(
                            raw_copy, format=raw_copy.format or "PNG")
                        if include_base64_images
                        else None
                    )
                    extracted_images.append(
                        ExtractedImage(
                            index=0,
                            caption=file_name,
                            label="source_image",
                            page_no=1,
                            bbox=None,
                            width=raw_copy.width,
                            height=raw_copy.height,
                            format=raw_copy.format or "PNG",
                            image=raw_copy,
                            base64_uri=b64_uri,
                            saved_path=str(path),
                        )
                    )
            except Exception as e:
                logger.debug(f"Could not load source image file directly: {e}")

    # --------------------------------------------------------------------------
    # 4. Extract Mathematical Formulas (LaTeX)
    # --------------------------------------------------------------------------
    extracted_formulas: List[ExtractedFormula] = []
    formula_idx = 0

    # A) Extract formulas detected during element iteration
    for elem in elements:
        if elem.type == "formula" and elem.text.strip():
            f_obj = ExtractedFormula(
                index=formula_idx,
                latex=elem.text.strip(),
                context=None,
                page_no=elem.page_no,
                bbox=elem.bbox,
            )
            f_obj.generate_summary()
            extracted_formulas.append(f_obj)
            formula_idx += 1

    # B) Regex scan markdown content for LaTeX blocks ($$ ... $$ or \[ ... \]) if not captured by docling
    if len(extracted_formulas) == 0 and markdown_content:
        latex_matches = re.finditer(
            r"\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]", markdown_content)
        for m in latex_matches:
            f_text = (m.group(1) or m.group(2) or "").strip()
            if f_text:
                f_obj = ExtractedFormula(
                    index=formula_idx,
                    latex=f"$${f_text}$$",
                    context=None,
                    page_no=1,
                    bbox=None,
                )
                f_obj.generate_summary()
                extracted_formulas.append(f_obj)
                formula_idx += 1

    # Ensure all tables and images have summaries generated
    for tbl in extracted_tables:
        tbl.generate_summary()
    for img in extracted_images:
        img.generate_summary()

    # Document statistics & metadata
    num_pages = 1
    try:
        if callable(getattr(doc, "num_pages", None)):
            num_pages = max(1, int(doc.num_pages()))
        elif hasattr(doc, "num_pages"):
            num_pages = max(1, int(getattr(doc, "num_pages", 1)))
        elif hasattr(doc, "pages"):
            num_pages = max(1, len(doc.pages))
    except Exception:
        num_pages = 1
    raw_chars = len(markdown_content or plain_text or "")
    est_tokens = max(1, max(int(len((markdown_content or plain_text).split()) * 1.3), raw_chars // 4)) if raw_chars > 0 else 0
    metadata: Dict[str, Any] = {
        "num_pages": num_pages,
        "num_elements": len(elements),
        "num_tables": len(extracted_tables),
        "num_images": len(extracted_images),
        "num_formulas": len(extracted_formulas),
        "file_extension": file_ext,
        "mime_type": mime_type,
        "raw_text_chars": raw_chars,
        "estimated_tokens": est_tokens,
    }

    logger.info(
        f"Successfully loaded '{file_name}': {len(elements)} elements, "
        f"{len(extracted_tables)} tables, {len(extracted_images)} images, "
        f"{len(extracted_formulas)} formulas, {num_pages} pages, ~{est_tokens} estimated tokens."
    )

    return MultimodalDocument(
        file_path=str(path) if path is not None else file_path,
        file_name=file_name,
        file_extension=file_ext,
        file_size_bytes=file_size,
        mime_type=mime_type,
        num_pages=num_pages,
        markdown_content=markdown_content,
        plain_text=plain_text,
        elements=elements,
        tables=extracted_tables,
        images=extracted_images,
        formulas=extracted_formulas,
        metadata=metadata,
        raw_document=doc,
    )


# Convenient aliases
load_document = load_file
load_multimodal_document = load_file


# ==============================================================================
# DIRECTORY SCANNING & RECURSIVE INGESTION UTILITIES
# ==============================================================================


def scan_directory(
    directory_path: Union[str, Path],
    recursive: bool = True,
    extensions: Optional[set[str]] = None,
) -> List[Path]:
    """Scan a directory for supported document, image, and data files.

    Args:
        directory_path: Root folder path to inspect.
        recursive: Whether to traverse nested child folders.
        extensions: Optional whitelist of extensions (defaults to SUPPORTED_DOCUMENT_EXTENSIONS).

    Returns:
        List of Path objects for all matching files, sorted deterministically.
    """
    root = Path(directory_path).resolve()
    if not root.exists() or not root.is_dir():
        logger.warning(
            f"Scan directory does not exist or is not a directory: {root}")
        return []

    target_exts = {e.lower()
                   for e in (extensions or SUPPORTED_DOCUMENT_EXTENSIONS)}
    matched_files: List[Path] = []

    pattern = "**/*" if recursive else "*"
    for item in root.glob(pattern):
        if item.is_file() and item.suffix.lower() in target_exts:
            matched_files.append(item)

    matched_files.sort(key=lambda p: str(p).lower())
    logger.info(
        f"Found {len(matched_files)} document(s) in '{root}' (recursive={recursive})"
    )
    return matched_files


def load_directory(
    directory_path: Union[str, Path],
    recursive: bool = True,
    extensions: Optional[set[str]] = None,
    ignore_errors: bool = True,
    **load_kwargs: Any,
) -> Iterator[MultimodalDocument]:
    """Recursively scan a directory and load all supported documents.

    Args:
        directory_path: Target directory to read documents from.
        recursive: Whether to scan subdirectories.
        extensions: Optional whitelist of extensions.
        ignore_errors: If True, log exceptions on unparseable files and continue.
        **load_kwargs: Parameters forwarded to load_file.

    Yields:
        MultimodalDocument for each successfully parsed file.
    """
    with _logfire_span(
        "loader.load_directory",
        directory=str(directory_path),
        recursive=recursive,
    ):
        files = scan_directory(
            directory_path, recursive=recursive, extensions=extensions)
        for file_path in files:
            try:
                doc = load_file(file_path, **load_kwargs)
                yield doc
            except Exception as e:
                if ignore_errors:
                    logger.warning(
                        f"Skipping failed document '{file_path}': {e}")
                    continue
                raise

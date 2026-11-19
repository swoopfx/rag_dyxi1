"""Production-level structure-aware document chunker using Docling and HybridChunker.

Architecture & Scope:
- Scope: Exclusively focuses on high-throughput, structure-aware document chunking using Docling.
  Embedding generation is decoupled and managed by app.services.retreival.embeddings.
  Both modules are composed in the downstream ingestion processor (app.ingestion.processor).
- Gemini Embedding 2 Tokenization:
  Tailored for Google Gemini Embedding 2 ('gemini-embedding-2'). Uses a native, zero-dependency
  GeminiTokenEstimator implementing Docling's BaseTokenizer protocol. This eliminates the need
  for external HuggingFace or OpenAI tokenizer downloads, avoiding network latency and HF Hub warnings.
- Decoupled Parallel Page Batching:
  * Decouples parallel page conversion from structure-aware chunking.
  * Slices large multi-hundred-page PDFs into bounded page batches (e.g., 25-50 pages).
  * Concurrently processes page batches via ThreadedPdfPipelineOptions and convert_all().
  * Streams converted sub-documents directly into HybridChunker without calling
    memory-heavy DoclingDocument.concatenate().
- Pipeline Threading & Batching:
  * Uses ThreadedPdfPipelineOptions with tuned layout_batch_size, ocr_batch_size, and table_batch_size.
  * Configures document_timeout to capture partial results and guard against pipeline hangs on malformed files.
- Performance Optimizations:
  * Disables unnecessary image generation (generate_page_images=False, generate_picture_images=False,
    images_scale=1.0) to eliminate rasterization overhead.
  * Skips OCR on digital PDFs: automatically probes the text layer and only enables OCR when scanned
    content is detected.
- Structure-Aware Contextualization:
  * Employs Docling's HybridChunker with merge_peers=True to merge consecutive undersized nodes
    (e.g., short bullet points or list items) under the same parent heading.
  * Injects hierarchical section headers and captions into contextualized_text via chunker.contextualize(chunk)
    for downstream vector embedding without requiring expensive LLM summarization.
- Interoperability with load.py:
  * Directly processes MultimodalDocument containers returned by load.py, preserving document metadata,
    extracted tables, and images.
"""

from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

from loguru import logger
import pypdf
from pydantic import BaseModel, ConfigDict, Field

# Docling imports
from docling.chunking import HybridChunker
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions
from docling.datamodel.settings import settings as docling_settings
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
from docling_core.types.doc import DoclingDocument
from docling_core.types.io import DocumentStream

# Project imports
from app.config import settings
from app.ingestion.loader.load import MultimodalDocument

# Logfire Observability Setup
try:
    import logfire

    _logfire_token = getattr(settings, "LOGFIRE_TOKEN",
                             None) or os.getenv("LOGFIRE_TOKEN")
    logfire.configure(
        send_to_logfire="if-token-present",
        token=_logfire_token,
        service_name="rag-chunking-service",
    )
except ImportError:
    logfire = None


class _NoopSpan:
    """No-op span placeholder for when Logfire is unavailable."""

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
        with logfire.span(name, **kwargs) as span:
            yield span
    else:
        yield _NOOP_SPAN


# ==============================================================================
# CONFIGURATION DEFAULTS
# ==============================================================================

# Sliced page batch size for large PDFs (range: 25-50 pages)
DEFAULT_PAGE_CHUNK_SIZE: int = 30
DEFAULT_DOC_BATCH_CONCURRENCY: int = 4  # Concurrency level for convert_all()
# Default max token budget per chunk (Gemini Embedding 2 supports up to 2048)
DEFAULT_MAX_TOKENS: int = 1200
# Layout model batch size (tune higher on GPU)
DEFAULT_LAYOUT_BATCH_SIZE: int = 4
DEFAULT_OCR_BATCH_SIZE: int = 4  # OCR model batch size (tune higher on GPU)
# Table recognition batch size (tune higher on GPU)
DEFAULT_TABLE_BATCH_SIZE: int = 4
# Timeout in seconds to prevent workers hanging on malformed files
DEFAULT_DOCUMENT_TIMEOUT: float = 120.0
# Minimum average characters/page to classify PDF as digital
MIN_DIGITAL_TEXT_CHARS: int = 40


# ==============================================================================
# NATIVE GEMINI TOKEN ESTIMATOR (ZERO EXTERNAL TOKENIZER REQUIRED)
# ==============================================================================


class GeminiTokenEstimator(BaseTokenizer):
    """Zero-dependency token estimator tailored for Gemini Embedding 2.

    What:
        Implements Docling's BaseTokenizer protocol without downloading, loading,
        or depending on external HuggingFace or OpenAI tokenizer models. Estimates
        token counts using standard word and character heuristic boundaries (~4 chars/token).

    Why:
        Gemini Embedding 2 ('gemini-embedding-2') performs its own tokenization remotely
        within Google's infrastructure. Downloading external HuggingFace models
        (e.g., sentence-transformers/all-MiniLM) introduces unnecessary network latency,
        HuggingFace Hub rate-limiting warnings, and bloated runtime dependencies.
        This native estimator satisfies Docling's BaseTokenizer contract locally,
        enforcing token limits and chunk boundaries with zero network overhead.
    """

    max_tokens: int = Field(
        default=DEFAULT_MAX_TOKENS, description="Maximum token budget allocated per chunk"
    )

    def count_tokens(self, text: str) -> int:
        """Estimate token count for given text using word and character heuristics.

        What:
            Calculates token count based on the maximum of (word_count * 1.3) and
            (char_count // 4).

        Why:
            Gemini's SentencePiece/BPE tokenizer averages ~4 characters per token across
            English text, and ~1.3 tokens per whitespace-separated word for mixed
            alphanumeric and technical vocabulary. This heuristic bounds chunk size reliably
            without requiring heavy tokenizer models.
        """
        if not text:
            return 0
        words = len(text.split())
        chars = len(text)
        return max(1, max(int(words * 1.3), chars // 4))

    def get_max_tokens(self) -> int:
        """Return the maximum token budget."""
        return self.max_tokens

    def get_tokenizer(self) -> Any:
        """Return self as the underlying tokenizer object."""
        return self


# ==============================================================================
# STRUCTURED CHUNK DATA MODELS
# ==============================================================================


class ChunkMetadata(BaseModel):
    """Detailed structural and provenance metadata for an individual chunk."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_file: Optional[str] = Field(
        default=None, description="Path or identifier of the source file")
    page_numbers: List[int] = Field(
        default_factory=list, description="Original 1-based page numbers spanning this chunk"
    )
    headings: List[str] = Field(
        default_factory=list, description="Hierarchical heading breadcrumbs leading to this chunk"
    )
    captions: List[str] = Field(
        default_factory=list, description="Captions associated with figures or tables in this chunk"
    )
    num_tokens: Optional[int] = Field(
        default=None, description="Estimated token count computed by the token estimator"
    )
    doc_items_count: int = Field(
        default=0, description="Number of distinct Docling document items merged into this chunk"
    )
    is_contextualized: bool = Field(
        default=True, description="Whether contextualized_text contains prepended hierarchical headers"
    )
    extra: Dict[str, Any] = Field(
        default_factory=dict, description="Additional custom metadata from loader or caller"
    )


class ProcessedChunk(BaseModel):
    """Production-grade document chunk generated by Docling HybridChunker.

    What:
        Contains both the raw extracted text and the hierarchically contextualized text,
        along with structural breadcrumbs (headings, captions), page provenance, and token counts.

    Why:
        Preserves raw text for concise LLM context display while providing enriched
        contextualized_text (via chunker.contextualize()) for high-recall vector embeddings
        in app.services.retreival.embeddings.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    chunk_id: str = Field(...,
                          description="Unique deterministic identifier for the chunk")
    index: int = Field(...,
                       description="Sequential order index of chunk within document")
    text: str = Field(...,
                      description="Raw text of the chunk without prepended context")
    contextualized_text: str = Field(
        ...,
        description="Hierarchically enriched text with section headers and captions prepended by HybridChunker",
    )
    metadata: ChunkMetadata = Field(...,
                                    description="Structural metadata and provenance")

    @property
    def embedding_text(self) -> str:
        """Text optimized for vector embeddings in app.services.retreival.embeddings.

        What:
            Returns contextualized_text containing prepended section headers and captions.

        Why:
            Injecting hierarchical section headers automatically resolves ambiguous pronouns
            and sub-points (e.g. '3. Eligibility' under 'Chapter 2: Health Insurance') without
            requiring costly LLM-based summarization calls.
        """
        return self.contextualized_text if self.contextualized_text.strip() else self.text

    @property
    def page_numbers(self) -> List[int]:
        """Convenience accessor for page numbers."""
        return self.metadata.page_numbers

    @property
    def headings(self) -> List[str]:
        """Convenience accessor for heading breadcrumbs."""
        return self.metadata.headings

    def to_langchain_document(self) -> Any:
        """Convert chunk to a standard LangChain Document object."""
        from langchain_core.documents import Document

        return Document(
            page_content=self.embedding_text,
            metadata={
                "chunk_id": self.chunk_id,
                "index": self.index,
                "raw_text": self.text,
                "source": self.metadata.source_file,
                "page_numbers": self.metadata.page_numbers,
                "headings": self.metadata.headings,
                "captions": self.metadata.captions,
                "num_tokens": self.metadata.num_tokens,
                "doc_items_count": self.metadata.doc_items_count,
                "is_contextualized": self.metadata.is_contextualized,
                **self.metadata.extra,
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to a JSON-serializable dictionary."""
        return self.model_dump()


# ==============================================================================
# PIPELINE CONFIGURATION & UTILITIES
# ==============================================================================


def is_digital_pdf(
    source: Union[str, Path, io.BytesIO],
    sample_pages: int = 5,
    min_chars_per_page: int = MIN_DIGITAL_TEXT_CHARS,
) -> bool:
    """Detect whether a PDF has an existing digital text layer or consists of scanned images.

    What:
        Inspects up to sample_pages of the PDF using pypdf. If the average extracted text
        length per page exceeds min_chars_per_page, the document is classified as a digital PDF.

    Why:
        Optical Character Recognition (OCR) is the primary CPU and GPU bottleneck during
        document parsing. Digital PDFs already contain clean vector text streams; running OCR
        on digital PDFs degrades throughput by 5x-10x with zero accuracy benefit. Skipping OCR
        when digital text is detected yields massive performance gains.
    """
    try:
        reader = pypdf.PdfReader(source)
        total_pages = len(reader.pages)
        if total_pages == 0:
            return False

        pages_to_check = min(total_pages, max(1, sample_pages))
        total_chars = 0
        for i in range(pages_to_check):
            page_text = reader.pages[i].extract_text() or ""
            total_chars += len(page_text.strip())

        avg_chars = total_chars / pages_to_check
        is_digital = avg_chars >= min_chars_per_page
        logger.debug(
            f"PDF digital text probe: {avg_chars:.1f} avg chars/page across {pages_to_check} pages -> "
            f"{'Digital (Skip OCR)' if is_digital else 'Scanned (Enable OCR)'}"
        )
        return is_digital
    except Exception as e:
        logger.warning(
            f"Could not probe PDF digital text layer: {e}. Defaulting to digital.")
        return True


def build_threaded_pdf_options(
    do_ocr: Optional[bool] = None,
    layout_batch_size: int = DEFAULT_LAYOUT_BATCH_SIZE,
    ocr_batch_size: int = DEFAULT_OCR_BATCH_SIZE,
    table_batch_size: int = DEFAULT_TABLE_BATCH_SIZE,
    document_timeout: float = DEFAULT_DOCUMENT_TIMEOUT,
    do_table_structure: bool = True,
    generate_page_images: bool = False,
    generate_picture_images: bool = False,
    images_scale: float = 1.0,
) -> ThreadedPdfPipelineOptions:
    """Build high-throughput ThreadedPdfPipelineOptions for production PDF conversion.

    What:
        Initializes Docling's ThreadedPdfPipelineOptions with tuned batch processing parameters,
        disables unnecessary image rendering, configures execution timeout, and conditionally
        disables OCR for digital documents.

    Why:
        1. Threaded Pipeline Batching: ThreadedPdfPipelineOptions parallelizes stage execution
           (layout, OCR, table structure) across thread pools. Tuning batch sizes (e.g. 4-8)
           maximizes CPU and GPU hardware utilization.
        2. Disabled Image Overhead: Setting generate_page_images=False and generate_picture_images=False
           eliminates raster image rendering overhead, saving substantial RAM and cutting conversion time in half.
        3. Hang Protection: Setting document_timeout captures partial results if a complex or
           corrupted document encounters an infinite loop or deep recursion in parser models.
    """
    opts = ThreadedPdfPipelineOptions()

    # Batch sizes for pipeline execution stages
    opts.layout_batch_size = max(1, layout_batch_size)
    opts.ocr_batch_size = max(1, ocr_batch_size)
    opts.table_batch_size = max(1, table_batch_size)

    # Edge-case timeout to prevent workers hanging on malformed files
    opts.document_timeout = max(5.0, document_timeout)

    # Disable expensive image generation unless multimodal captioning is explicitly required
    opts.generate_page_images = generate_page_images
    opts.generate_picture_images = generate_picture_images
    opts.images_scale = images_scale

    # Structural feature flags
    opts.do_table_structure = do_table_structure
    opts.do_ocr = bool(do_ocr) if do_ocr is not None else False

    return opts


def build_hybrid_chunker(
    max_tokens: int = DEFAULT_MAX_TOKENS,
    merge_peers: bool = True,
    custom_tokenizer: Optional[BaseTokenizer] = None,
) -> HybridChunker:
    """Initialize Docling HybridChunker with Gemini token estimator and peer merging.

    What:
        Instantiates HybridChunker using GeminiTokenEstimator (or an optional custom BaseTokenizer)
        with merge_peers=True.

    Why:
        1. merge_peers=True: Merges consecutive undersized nodes (e.g. short list items, bullet points)
           that share the same parent heading into a single unified chunk, preventing vector DB bloat
           and reducing total embedding API calls.
        2. Native Token Bound: Uses GeminiTokenEstimator so no external HuggingFace models are required.
    """
    tok = custom_tokenizer or GeminiTokenEstimator(max_tokens=max_tokens)
    return HybridChunker(
        tokenizer=tok,
        merge_peers=merge_peers,
    )


# ==============================================================================
# PDF PARALLEL PAGE SLICING UTILITIES
# ==============================================================================


def slice_pdf_to_page_batches(
    file_path: Union[str, Path],
    page_chunk_size: int = DEFAULT_PAGE_CHUNK_SIZE,
) -> List[tuple[int, int, DocumentStream]]:
    """Slice a large PDF into in-memory DocumentStream batches of page_chunk_size.

    What:
        Reads the target PDF with pypdf. If total pages exceed page_chunk_size, segments
        the document into slices of page_chunk_size pages (e.g., 25-50 pages). Each slice is
        written to an in-memory BytesIO buffer and wrapped in a DocumentStream.

    Why:
        Docling's DocumentConverter processes documents sequentially when given a single file path.
        Slicing a massive PDF into parallel sub-documents allows convert_all() to dispatch
        independent page batches across multiple worker threads concurrently, drastically
        reducing ingestion time for 100+ page technical manuals and reports.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"PDF file does not exist: {path}")

    reader = pypdf.PdfReader(str(path))
    total_pages = len(reader.pages)
    stem = path.stem

    # If small enough, return as a single DocumentStream without slicing overhead
    if total_pages <= page_chunk_size:
        stream = DocumentStream(name=path.name, stream=open(path, "rb"))
        return [(1, total_pages, stream)]

    batches: List[tuple[int, int, DocumentStream]] = []
    chunk_size = max(1, page_chunk_size)

    for start_idx in range(0, total_pages, chunk_size):
        end_idx = min(start_idx + chunk_size, total_pages)
        writer = pypdf.PdfWriter()
        for p in range(start_idx, end_idx):
            writer.add_page(reader.pages[p])

        buf = io.BytesIO()
        writer.write(buf)
        buf.seek(0)

        # 1-based page numbers for human-readable logging and metadata
        start_page = start_idx + 1
        end_page = end_idx
        batch_name = f"{stem}#pages_{start_page}_{end_page}.pdf"
        stream = DocumentStream(name=batch_name, stream=buf)
        batches.append((start_page, end_page, stream))

    logger.info(
        f"Sliced '{path.name}' ({total_pages} pages) into {len(batches)} parallel batches "
        f"of up to {page_chunk_size} pages each."
    )
    return batches


# ==============================================================================
# CORE CHUNKING ENGINES
# ==============================================================================


def chunk_docling_document(
    dl_doc: DoclingDocument,
    chunker: Optional[HybridChunker] = None,
    page_offset: int = 0,
    source_file: Optional[str] = None,
    chunk_index_offset: int = 0,
    extra_metadata: Optional[Dict[str, Any]] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    merge_peers: bool = True,
) -> List[ProcessedChunk]:
    """Execute structure-aware chunking on a DoclingDocument using HybridChunker.

    What:
        Applies HybridChunker to the document, extracts raw text and contextualized text
        (via chunker.contextualize(chunk)), extracts page provenance and heading breadcrumbs,
        and constructs ProcessedChunk objects.

    Why:
        Decouples the chunking operation from file conversion, allowing already converted
        sub-documents or MultimodalDocument objects to be chunked independently without
        re-parsing the source files.
    """
    if chunker is None:
        chunker = build_hybrid_chunker(
            max_tokens=max_tokens, merge_peers=merge_peers)

    doc_name = getattr(dl_doc, "name", "document") or "document"
    file_id = Path(source_file).name if source_file else doc_name

    chunks: List[ProcessedChunk] = []
    idx = chunk_index_offset

    with _logfire_span("hybrid_chunker.chunk", doc_name=doc_name, page_offset=page_offset):
        for raw_chunk in chunker.chunk(dl_doc):
            # 1. Raw and Contextualized Text
            # contextualize() automatically prepends hierarchical headers and captions
            raw_text = raw_chunk.text or ""
            try:
                contextualized_text = chunker.contextualize(
                    raw_chunk) or raw_text
            except Exception as e:
                logger.debug(
                    f"contextualize() failed on chunk {idx}: {e}. Using raw text.")
                contextualized_text = raw_text

            # 2. Extract Headings Breadcrumbs
            headings: List[str] = []
            if hasattr(raw_chunk, "meta") and getattr(raw_chunk.meta, "headings", None):
                headings = [str(h)
                            for h in raw_chunk.meta.headings if str(h).strip()]

            # 3. Extract Captions
            captions: List[str] = []
            if hasattr(raw_chunk, "meta") and getattr(raw_chunk.meta, "captions", None):
                captions = [str(c)
                            for c in raw_chunk.meta.captions if str(c).strip()]

            # 4. Extract Page Numbers from Item Provenance
            # Adjust by page_offset if this document is a sliced sub-document batch
            page_set = set()
            doc_items_count = 0
            if hasattr(raw_chunk, "meta") and getattr(raw_chunk.meta, "doc_items", None):
                doc_items_count = len(raw_chunk.meta.doc_items)
                for item in raw_chunk.meta.doc_items:
                    prov_list = getattr(item, "prov", None) or []
                    for p in prov_list:
                        p_no = getattr(p, "page_no", None)
                        if p_no is not None:
                            page_set.add(int(p_no) + page_offset)

            sorted_pages = sorted(list(page_set))

            # 5. Token Count Estimation via GeminiTokenEstimator
            num_tokens = None
            if hasattr(chunker, "tokenizer") and hasattr(chunker.tokenizer, "count_tokens"):
                try:
                    num_tokens = chunker.tokenizer.count_tokens(
                        contextualized_text)
                except Exception:
                    pass

            # 6. Build Structured Metadata
            meta = ChunkMetadata(
                source_file=source_file or doc_name,
                page_numbers=sorted_pages,
                headings=headings,
                captions=captions,
                num_tokens=num_tokens,
                doc_items_count=doc_items_count,
                is_contextualized=bool(contextualized_text != raw_text),
                extra=extra_metadata or {},
            )

            chunk_id = f"{file_id}#chunk_{idx:05d}"
            chunks.append(
                ProcessedChunk(
                    chunk_id=chunk_id,
                    index=idx,
                    text=raw_text,
                    contextualized_text=contextualized_text,
                    metadata=meta,
                )
            )
            idx += 1

    return chunks


# ==============================================================================
# STREAMING DECOUPLED PARALLEL CONVERTER & CHUNKER
# ==============================================================================


def stream_convert_and_chunk(
    source: Union[str, Path, io.BytesIO, DocumentStream],
    page_chunk_size: int = DEFAULT_PAGE_CHUNK_SIZE,
    doc_batch_concurrency: int = DEFAULT_DOC_BATCH_CONCURRENCY,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    merge_peers: bool = True,
    do_ocr: Optional[bool] = None,
    layout_batch_size: int = DEFAULT_LAYOUT_BATCH_SIZE,
    ocr_batch_size: int = DEFAULT_OCR_BATCH_SIZE,
    table_batch_size: int = DEFAULT_TABLE_BATCH_SIZE,
    document_timeout: float = DEFAULT_DOCUMENT_TIMEOUT,
    converter: Optional[DocumentConverter] = None,
    chunker: Optional[HybridChunker] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Iterator[ProcessedChunk]:
    """Streamingly convert and structure-chunk a document using parallel page batches.

    What:
        1. Slices large PDFs into parallel page batches (e.g., 25-50 pages).
        2. Configures DocumentConverter with ThreadedPdfPipelineOptions and sets
           doc_batch_concurrency to parallelize conversion across workers.
        3. Calls convert_all() to stream conversion results back as each batch finishes.
        4. Immediately streams each converted sub-document into HybridChunker without
           calling DoclingDocument.concatenate().
        5. Yields ProcessedChunk instances in real-time as chunks are generated.

    Why:
        1. Memory Efficiency: Calling DoclingDocument.concatenate() on multi-hundred-page
           documents incurs massive memory allocations and object graph traversals. Streaming
           each sub-document straight into HybridChunker maintains a flat memory profile.
        2. Progressive Stream: Downstream ingestion steps can begin vectorizing or processing
           the first page batch immediately without waiting for page 500 to finish.
        3. Fault Isolation: If a corrupt element in batch 4 triggers a timeout or partial failure,
           earlier batches (1-3) have already been safely chunked and emitted.
    """
    with _logfire_span(
        "chunker.stream_convert_and_chunk",
        source=str(source) if not isinstance(
            source, io.BytesIO) else "BytesIO",
        page_chunk_size=page_chunk_size,
        doc_batch_concurrency=doc_batch_concurrency,
        max_tokens=max_tokens,
    ):
        # 1. Resolve source path and type
        source_path: Optional[Path] = None
        is_pdf = False
        source_name = "document"

        if isinstance(source, (str, Path)):
            p = Path(source).resolve()
            if p.exists() and p.is_file():
                source_path = p
                source_name = p.name
                is_pdf = p.suffix.lower() == ".pdf"
        elif isinstance(source, DocumentStream):
            source_name = source.name
            is_pdf = source.name.lower().endswith(".pdf")

        # 2. Determine OCR necessity (skip OCR on digital PDFs)
        active_do_ocr = do_ocr
        if active_do_ocr is None and is_pdf and source_path is not None:
            active_do_ocr = not is_digital_pdf(source_path)

        # 3. Initialize Threaded Converter if not supplied
        if converter is None:
            pipeline_opts = build_threaded_pdf_options(
                do_ocr=active_do_ocr,
                layout_batch_size=layout_batch_size,
                ocr_batch_size=ocr_batch_size,
                table_batch_size=table_batch_size,
                document_timeout=document_timeout,
                generate_page_images=False,
                generate_picture_images=False,
                images_scale=1.0,
            )
            format_options = {
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts),
            }
            converter = DocumentConverter(format_options=format_options)

        # 4. Initialize HybridChunker with GeminiTokenEstimator if not supplied
        if chunker is None:
            chunker = build_hybrid_chunker(
                max_tokens=max_tokens,
                merge_peers=merge_peers,
            )

        # 5. Configure Docling's concurrent batch settings for convert_all()
        docling_settings.perf.doc_batch_concurrency = max(
            1, doc_batch_concurrency)
        docling_settings.perf.doc_batch_size = max(
            docling_settings.perf.doc_batch_concurrency, 2)

        # 6. Prepare sliced DocumentStreams for large PDFs or single source
        page_offsets: Dict[str, int] = {}
        input_sources: List[Union[Path, DocumentStream]] = []

        if is_pdf and source_path is not None:
            try:
                sliced_batches = slice_pdf_to_page_batches(
                    source_path, page_chunk_size=page_chunk_size
                )
                for start_p, end_p, stream in sliced_batches:
                    input_sources.append(stream)
                    # Provenance page_no in a sliced sub-doc is 1-based from slice start;
                    # offset maps it back to original document page number: (start_p - 1)
                    page_offsets[stream.name] = start_p - 1
            except Exception as e:
                logger.warning(
                    f"Failed to slice PDF '{source_path}': {e}. Processing as single document."
                )
                input_sources = [source_path]
                page_offsets[source_path.name] = 0
        else:
            if isinstance(source, (str, Path)):
                input_sources = [Path(source)]
                page_offsets[Path(source).name] = 0
            elif isinstance(source, DocumentStream):
                input_sources = [source]
                page_offsets[source.name] = 0

        # 7. Execute parallel conversion and stream results into HybridChunker
        chunk_idx = 0
        logger.info(
            f"Converting and chunking {len(input_sources)} stream batch(es) "
            f"with concurrency={doc_batch_concurrency}..."
        )

        for conv_res in converter.convert_all(input_sources, raises_on_error=False):
            if conv_res.status not in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS):
                logger.warning(
                    f"Conversion error on '{conv_res.input.file.name}' with status: {conv_res.status}. "
                    f"Errors: {[err.error_message for err in conv_res.errors]}"
                )
                continue

            sub_doc: DoclingDocument = conv_res.document
            doc_key = conv_res.input.file.name
            offset = page_offsets.get(doc_key, 0)

            # Stream each converted sub-document straight into HybridChunker
            batch_chunks = chunk_docling_document(
                dl_doc=sub_doc,
                chunker=chunker,
                page_offset=offset,
                source_file=str(source_path) if source_path else source_name,
                chunk_index_offset=chunk_idx,
                extra_metadata=extra_metadata,
            )

            for chunk in batch_chunks:
                yield chunk
                chunk_idx += 1


# ==============================================================================
# MAIN CHUNKING ENTRY POINTS
# ==============================================================================


def chunk_text(
    text: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    source_file: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> List[ProcessedChunk]:
    """Chunk raw text or markdown into ProcessedChunk objects using paragraph boundaries and token estimation.

    Args:
        text: Raw text or markdown string to chunk.
        max_tokens: Maximum token budget per chunk.
        source_file: Identifier or filename of the source file.
        extra_metadata: Additional metadata dictionary to attach to chunks.

    Returns:
        List of ProcessedChunk objects with estimated token counts and sequential IDs.
    """
    if not text or not text.strip():
        return []

    estimator = GeminiTokenEstimator(max_tokens=max_tokens)
    paragraphs = text.split("\n\n")
    chunks: List[ProcessedChunk] = []

    current_paragraphs: List[str] = []
    current_tokens = 0
    idx = 0
    file_id = Path(source_file).name if source_file else "document"

    for para in paragraphs:
        p_str = para.strip()
        if not p_str:
            continue
        p_tokens = estimator.count_tokens(p_str)

        # Handle individual paragraph that exceeds max_tokens on its own
        if p_tokens > max_tokens:
            if current_paragraphs:
                chunk_text_str = "\n\n".join(current_paragraphs)
                chunk_id = f"{file_id}#chunk_{idx:05d}"
                chunks.append(
                    ProcessedChunk(
                        chunk_id=chunk_id,
                        index=idx,
                        text=chunk_text_str,
                        contextualized_text=chunk_text_str,
                        metadata=ChunkMetadata(
                            source_file=source_file or file_id,
                            page_numbers=[1],
                            headings=[],
                            captions=[],
                            num_tokens=current_tokens,
                            doc_items_count=len(current_paragraphs),
                            is_contextualized=False,
                            extra=extra_metadata or {},
                        ),
                    )
                )
                idx += 1
                current_paragraphs = []
                current_tokens = 0

            # Sub-split long paragraph by words
            words = p_str.split()
            sub_words: List[str] = []
            for w in words:
                sub_words.append(w)
                if estimator.count_tokens(" ".join(sub_words)) >= max_tokens:
                    sub_str = " ".join(sub_words)
                    chunk_id = f"{file_id}#chunk_{idx:05d}"
                    chunks.append(
                        ProcessedChunk(
                            chunk_id=chunk_id,
                            index=idx,
                            text=sub_str,
                            contextualized_text=sub_str,
                            metadata=ChunkMetadata(
                                source_file=source_file or file_id,
                                page_numbers=[1],
                                headings=[],
                                captions=[],
                                num_tokens=estimator.count_tokens(sub_str),
                                doc_items_count=1,
                                is_contextualized=False,
                                extra=extra_metadata or {},
                            ),
                        )
                    )
                    idx += 1
                    sub_words = []
            if sub_words:
                sub_str = " ".join(sub_words)
                current_paragraphs.append(sub_str)
                current_tokens += estimator.count_tokens(sub_str)
            continue

        if current_tokens + p_tokens > max_tokens and current_paragraphs:
            chunk_text_str = "\n\n".join(current_paragraphs)
            chunk_id = f"{file_id}#chunk_{idx:05d}"
            chunks.append(
                ProcessedChunk(
                    chunk_id=chunk_id,
                    index=idx,
                    text=chunk_text_str,
                    contextualized_text=chunk_text_str,
                    metadata=ChunkMetadata(
                        source_file=source_file or file_id,
                        page_numbers=[1],
                        headings=[],
                        captions=[],
                        num_tokens=current_tokens,
                        doc_items_count=len(current_paragraphs),
                        is_contextualized=False,
                        extra=extra_metadata or {},
                    ),
                )
            )
            idx += 1
            current_paragraphs = [p_str]
            current_tokens = p_tokens
        else:
            current_paragraphs.append(p_str)
            current_tokens += p_tokens

    if current_paragraphs:
        chunk_text_str = "\n\n".join(current_paragraphs)
        chunk_id = f"{file_id}#chunk_{idx:05d}"
        chunks.append(
            ProcessedChunk(
                chunk_id=chunk_id,
                index=idx,
                text=chunk_text_str,
                contextualized_text=chunk_text_str,
                metadata=ChunkMetadata(
                    source_file=source_file or file_id,
                    page_numbers=[1],
                    headings=[],
                    captions=[],
                    num_tokens=current_tokens,
                    doc_items_count=len(current_paragraphs),
                    is_contextualized=False,
                    extra=extra_metadata or {},
                ),
            )
        )

    return chunks


def chunk_document(
    document: Union[MultimodalDocument, DoclingDocument, str, Path, io.BytesIO, DocumentStream],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    merge_peers: bool = True,
    page_chunk_size: int = DEFAULT_PAGE_CHUNK_SIZE,
    doc_batch_concurrency: int = DEFAULT_DOC_BATCH_CONCURRENCY,
    do_ocr: Optional[bool] = None,
    layout_batch_size: int = DEFAULT_LAYOUT_BATCH_SIZE,
    ocr_batch_size: int = DEFAULT_OCR_BATCH_SIZE,
    table_batch_size: int = DEFAULT_TABLE_BATCH_SIZE,
    document_timeout: float = DEFAULT_DOCUMENT_TIMEOUT,
    custom_tokenizer: Optional[BaseTokenizer] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> List[ProcessedChunk]:
    """Execute production-level structure-aware document chunking using Docling HybridChunker.

    What:
        Unified entry point for chunking. Accepts:
        1. MultimodalDocument instances from load.py (extracts raw_document and preserves loader metadata).
        2. Raw DoclingDocument instances.
        3. File paths (str / Path), URLs, or DocumentStreams (runs decoupled parallel conversion and chunking).

    Why:
        Provides a single, intuitive interface that integrates seamlessly with existing
        loaders (load.py) while also supporting direct end-to-end parallel conversion
        and chunking for high-scale ingestion pipelines.

    Args:
        document: MultimodalDocument from load.py, raw DoclingDocument, file path, or stream.
        max_tokens: Max tokens per chunk (default: 1200, tailored for Gemini Embedding 2).
        merge_peers: Whether to merge consecutive undersized items under the same heading (default: True).
        page_chunk_size: Number of pages per sliced PDF batch for large files (default: 30).
        doc_batch_concurrency: Number of concurrent worker threads for convert_all (default: 4).
        do_ocr: Force OCR on/off (None = auto-detect digital vs scanned PDF).
        layout_batch_size: Layout model batch size (default: 4).
        ocr_batch_size: OCR model batch size (default: 4).
        table_batch_size: Table recognition model batch size (default: 4).
        document_timeout: Pipeline conversion timeout in seconds (default: 120.0).
        custom_tokenizer: Optional custom BaseTokenizer (defaults to GeminiTokenEstimator).
        extra_metadata: Additional custom metadata dictionary attached to all chunks.

    Returns:
        List[ProcessedChunk]: Ordered list of structured, contextualized chunks.
    """
    with _logfire_span("chunk_document", max_tokens=max_tokens, merge_peers=merge_peers):
        # 1. Handle MultimodalDocument returned by load.py
        if isinstance(document, MultimodalDocument):
            logger.info(
                f"Chunking MultimodalDocument '{document.file_name}' using HybridChunker...")
            chunker = build_hybrid_chunker(
                max_tokens=max_tokens,
                merge_peers=merge_peers,
                custom_tokenizer=custom_tokenizer,
            )

            # Combine loader metadata with caller's extra metadata
            combined_meta = dict(document.metadata)
            if extra_metadata:
                combined_meta.update(extra_metadata)

            if document.raw_document is not None:
                return chunk_docling_document(
                    dl_doc=document.raw_document,
                    chunker=chunker,
                    page_offset=0,
                    source_file=document.file_path or document.file_name,
                    extra_metadata=combined_meta,
                    max_tokens=max_tokens,
                    merge_peers=merge_peers,
                )
            elif document.markdown_content or document.plain_text:
                # Direct text chunking for plain text/unstructured documents
                text_to_chunk = document.markdown_content or document.plain_text
                return chunk_text(
                    text=text_to_chunk,
                    max_tokens=max_tokens,
                    source_file=document.file_path or document.file_name,
                    extra_metadata=combined_meta,
                )
            else:
                # If raw_document was not retained, re-stream via file_path
                if document.file_path and Path(document.file_path).exists():
                    return list(
                        stream_convert_and_chunk(
                            source=document.file_path,
                            page_chunk_size=page_chunk_size,
                            doc_batch_concurrency=doc_batch_concurrency,
                            max_tokens=max_tokens,
                            merge_peers=merge_peers,
                            do_ocr=do_ocr,
                            layout_batch_size=layout_batch_size,
                            ocr_batch_size=ocr_batch_size,
                            table_batch_size=table_batch_size,
                            document_timeout=document_timeout,
                            chunker=chunker,
                            extra_metadata=combined_meta,
                        )
                    )
                raise ValueError(
                    "MultimodalDocument has no raw_document, text content, or accessible file_path to chunk."
                )

        # 2. Handle raw DoclingDocument instance directly
        if isinstance(document, DoclingDocument):
            chunker = build_hybrid_chunker(
                max_tokens=max_tokens,
                merge_peers=merge_peers,
                custom_tokenizer=custom_tokenizer,
            )
            return chunk_docling_document(
                dl_doc=document,
                chunker=chunker,
                page_offset=0,
                source_file=getattr(document, "name", "document"),
                extra_metadata=extra_metadata,
                max_tokens=max_tokens,
                merge_peers=merge_peers,
            )

        # 3. File path, URL, DocumentStream, or BytesIO -> decoupled streaming pipeline
        chunker = build_hybrid_chunker(
            max_tokens=max_tokens,
            merge_peers=merge_peers,
            custom_tokenizer=custom_tokenizer,
        )
        return list(
            stream_convert_and_chunk(
                source=document,
                page_chunk_size=page_chunk_size,
                doc_batch_concurrency=doc_batch_concurrency,
                max_tokens=max_tokens,
                merge_peers=merge_peers,
                do_ocr=do_ocr,
                layout_batch_size=layout_batch_size,
                ocr_batch_size=ocr_batch_size,
                table_batch_size=table_batch_size,
                document_timeout=document_timeout,
                chunker=chunker,
                extra_metadata=extra_metadata,
            )
        )


def stream_chunks(
    document: Union[MultimodalDocument, DoclingDocument, str, Path, io.BytesIO, DocumentStream],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    merge_peers: bool = True,
    page_chunk_size: int = DEFAULT_PAGE_CHUNK_SIZE,
    doc_batch_concurrency: int = DEFAULT_DOC_BATCH_CONCURRENCY,
    do_ocr: Optional[bool] = None,
    layout_batch_size: int = DEFAULT_LAYOUT_BATCH_SIZE,
    ocr_batch_size: int = DEFAULT_OCR_BATCH_SIZE,
    table_batch_size: int = DEFAULT_TABLE_BATCH_SIZE,
    document_timeout: float = DEFAULT_DOCUMENT_TIMEOUT,
    custom_tokenizer: Optional[BaseTokenizer] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Iterator[ProcessedChunk]:
    """Streamingly yield ProcessedChunk instances in real-time as page batches convert."""
    if isinstance(document, (MultimodalDocument, DoclingDocument)):
        for chunk in chunk_document(
            document=document,
            max_tokens=max_tokens,
            merge_peers=merge_peers,
            custom_tokenizer=custom_tokenizer,
            extra_metadata=extra_metadata,
        ):
            yield chunk
    else:
        yield from stream_convert_and_chunk(
            source=document,
            page_chunk_size=page_chunk_size,
            doc_batch_concurrency=doc_batch_concurrency,
            max_tokens=max_tokens,
            merge_peers=merge_peers,
            do_ocr=do_ocr,
            layout_batch_size=layout_batch_size,
            ocr_batch_size=ocr_batch_size,
            table_batch_size=table_batch_size,
            document_timeout=document_timeout,
            extra_metadata=extra_metadata,
        )


def chunk_multimodal_documents(
    documents: Sequence[MultimodalDocument],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    merge_peers: bool = True,
    **kwargs: Any,
) -> List[ProcessedChunk]:
    """Batch chunk multiple MultimodalDocument instances into a single unified list of chunks.

    Args:
        documents: Sequence of MultimodalDocument instances (from load_directory or load_file).
        max_tokens: Maximum token budget allocated per chunk.
        merge_peers: Whether to merge consecutive undersized items sharing the same parent heading.
        **kwargs: Additional parameters passed to chunk_document.

    Returns:
        Flattened list of all ProcessedChunk objects across all input documents.
    """
    all_chunks: List[ProcessedChunk] = []
    for doc in documents:
        doc_chunks = chunk_document(
            document=doc,
            max_tokens=max_tokens,
            merge_peers=merge_peers,
            **kwargs,
        )
        all_chunks.extend(doc_chunks)
    return all_chunks


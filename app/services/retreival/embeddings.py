"""Multimodal embeddings service using Google Gemini Embedding 2 and Jina Embeddings v4 fallback.

Features:
- Primary Model: Google Gemini Embedding 2 ('gemini-embedding-2') with native
  multimodal capabilities (text, images, and visual elements).
- Fallback Model: Jina AI universal multimodal model ('jina-embeddings-v4')
  with native image and text embeddings and Matryoshka Representation Learning (MRL)
  dimension truncation (default: 768), guaranteeing identical output dimensionality
  for vector database consistency.
- Proactive & Reactive Rate Limiting:
  * Thread-safe and async-safe Token Bucket rate limiter (configurable RPM).
  * 429 / ResourceExhausted detection with 'retry-after' delay extraction.
  * Shared throttling across concurrent worker threads to prevent cascading rate limits.
  * Automatic failover to Jina Embeddings v4 upon quota exhaustion or server errors.
- Automatic batching with configurable batch size and concurrency control.
- 1-to-1 vector generation: handles Gemini Embedding 2's aggregation behavior by
  concurrently embedding individual chunks in parallel batches.
- Dimension control via Matryoshka Representation Learning (MRL) (e.g. 768, 512, 256).
- Low latency via concurrent parallel requests and persistent client connection pooling.
- Synchronous and asynchronous APIs for text, image, and multimodal inputs.
"""

from __future__ import annotations

import asyncio
import base64
from contextlib import contextmanager
import io
import math
import mimetypes
import os
from pathlib import Path
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence, Union

from loguru import logger
from PIL import Image

try:
    from langchain_core.embeddings import Embeddings
except ImportError:
    class Embeddings:  # type: ignore
        pass

from app.config import settings

# Observability
from app.observability import observe, observe_block

_logfire_span = observe_block


# HTTP client for Jina AI API
try:
    import httpx
except ImportError:
    httpx = None  # type: ignore

# Google GenAI imports
try:
    from google import genai
    from google.genai import types
    from google.genai.errors import APIError, ClientError
except ImportError:
    genai = None  # type: ignore
    types = None  # type: ignore
    APIError = Exception  # type: ignore
    ClientError = Exception  # type: ignore


# ==============================================================================
# CONFIGURATION DEFAULTS
# ==============================================================================

DEFAULT_PRIMARY_MODEL: str = "gemini-embedding-2"
DEFAULT_FALLBACK_MODEL: str = "jina-embeddings-v4"
JINA_API_URL: str = "https://api.jina.ai/v1/embeddings"
DEFAULT_DIMENSION: int = 768
DEFAULT_BATCH_SIZE: int = 32
MAX_GOOGLE_BATCH_SIZE: int = 100
DEFAULT_MAX_CONCURRENCY: int = 8
DEFAULT_TIMEOUT_SECONDS: float = 30.0
DEFAULT_MAX_RETRIES: int = 3
DEFAULT_RPM_LIMIT: int = 600  # Requests per minute default (10 req/sec)
DOCUMENT_TASK_TYPE: str = "RETRIEVAL_DOCUMENT"
QUERY_TASK_TYPE: str = "RETRIEVAL_QUERY"

# Reusable client caches for latency optimization (connection pooling)
# Reusing persistent sessions eliminates TLS handshake overhead on high-frequency calls.
_GENAI_CLIENT: Optional[Any] = None
_JINA_CLIENT: Optional[Any] = None
_ASYNC_JINA_CLIENT: Optional[Any] = None
# Shared rate limiter instances keyed by RPM to coordinate rate-limiting across concurrent workers
_RATE_LIMITERS: Dict[int, "TokenBucketRateLimiter"] = {}
_RATE_LIMITER_LOCK = threading.Lock()

# Supported image input types
ImageInput = Union[Image.Image, bytes, bytearray, str, Path, Any]


# ==============================================================================
# TOKEN BUCKET RATE LIMITER (PROACTIVE RATE LIMITING)
# ==============================================================================


@observe()
class TokenBucketRateLimiter:
    """Thread-safe and async-safe Token Bucket Rate Limiter.

    What:
        Regulates the dispatch rate of outgoing embedding requests by maintaining
        a virtual bucket of tokens that replenishes continuously at a fixed rate
        (rpm / 60 tokens per second) up to a maximum burst capacity. Callers must
        acquire tokens before issuing requests to external embedding APIs.

    Why:
        Google Gemini and other remote embedding providers enforce strict Requests-Per-Minute
        (RPM) quotas. Without proactive client-side rate limiting, concurrent worker
        threads dispatching chunks in parallel would immediately saturate remote quotas,
        leading to cascading HTTP 429 (Too Many Requests) errors and failed ingestion jobs.
        The token bucket pattern absorbs short-lived concurrency spikes while ensuring
        sustained throughput stays strictly within quota limits.
    """

    def __init__(self, rpm: int = DEFAULT_RPM_LIMIT, burst_size: Optional[int] = None):
        """Initialize the token bucket with an RPM quota and burst allowance.

        What:
            Sets the continuous token accrual rate (tokens/second), calculates the
            maximum burst capacity, and initializes the bucket at full capacity.

        Why:
            Allowing a burst capacity of approximately 1/6th RPM (~10 seconds of quota)
            lets parallel worker threads dispatch initial slices concurrently without
            needlessly blocking, while preventing excessive bursts that could trigger
            upstream server-side DDoS/burst detection.
        """
        self.rpm = max(1, rpm)
        self.rate = self.rpm / 60.0  # tokens added per second
        # Burst capacity absorbs concurrent worker dispatch (~10s quota allowance) without tripping RPM
        self.capacity = burst_size or max(
            1, min(self.rpm, int(self.rpm / 6)))  # burst allowance
        self.tokens = float(self.capacity)
        self.last_update = time.monotonic()
        self._lock = threading.Lock()

    def _replenish(self, now: float) -> None:
        """Accrue new tokens based on monotonic time elapsed since the last update.

        What:
            Calculates newly generated tokens using (elapsed_time * rate) and clamps
            the total token balance to the configured bucket capacity.

        Why:
            Lazy evaluation on demand avoids running a dedicated background timer thread,
            saving system resources and eliminating thread synchronization overhead.
        """
        # Lazy replenishment: compute accrued tokens from elapsed time instead of background polling
        elapsed = now - self.last_update
        self.tokens = min(float(self.capacity),
                          self.tokens + elapsed * self.rate)
        self.last_update = now

    def acquire(self, tokens: int = 1) -> None:
        """Synchronously acquire tokens, blocking if the bucket is depleted.

        What:
            Checks for available tokens and deducts them if present. If insufficient,
            computes the exact sleep time required to accumulate the deficit and sleeps
            in short intervals (up to 1.0s) until tokens are secured.

        Why:
            Ensures synchronous worker threads in a ThreadPoolExecutor wait predictably
            for quota to replenish rather than dispatching requests that would fail with 429.
            Sleeping in bounded <=1.0s slices allows thread responsiveness and prevents drift.
        """
        while True:
            with self._lock:
                now = time.monotonic()
                self._replenish(now)
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return
                # Calculate sleep duration to refill deficit, sliced to <=1s intervals to check for dynamic updates
                needed = tokens - self.tokens
                sleep_time = needed / self.rate
            if sleep_time > 0.05 and logfire:
                logfire.info(
                    "Rate limiter wait: sleeping {sleep_time:.2f}s for {tokens} token(s)",
                    sleep_time=sleep_time,
                    tokens=tokens,
                    rpm=self.rpm,
                )
            time.sleep(min(sleep_time, 1.0))

    async def acquire_async(self, tokens: int = 1) -> None:
        """Asynchronously acquire tokens without blocking the asyncio event loop.

        What:
            Asynchronous counterpart to acquire(); computes token deficit and pauses
            the calling coroutine with asyncio.sleep() until tokens become available.

        Why:
            In async web applications (e.g. FastAPI endpoints), using time.sleep()
            blocks the entire event loop, freezing all concurrent requests.
            Using asyncio.sleep() yields execution back to the loop while quota regenerates.
        """
        while True:
            now = time.monotonic()
            with self._lock:
                self._replenish(now)
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return
                # Non-blocking async sleep yields execution while token balance accumulates
                needed = tokens - self.tokens
                sleep_time = needed / self.rate
            if sleep_time > 0.05 and logfire:
                logfire.info(
                    "Async rate limiter wait: sleeping {sleep_time:.2f}s for {tokens} token(s)",
                    sleep_time=sleep_time,
                    tokens=tokens,
                    rpm=self.rpm,
                )
            await asyncio.sleep(min(sleep_time, 1.0))

    def throttle(self, penalty_seconds: float) -> None:
        """Temporarily throttle the bucket by imposing a negative token deficit.

        What:
            Deducts tokens proportional to penalty_seconds * rate, driving the balance
            into negative debt and resetting the last update timestamp.

        Why:
            When an external service returns HTTP 429 despite proactive rate limiting,
            all active threads must immediately pause. Driving the token balance into debt
            forces all subsequent callers to wait until normal elapsed time pays off the debt,
            enforcing a coordinated cooldown across all worker threads.
        """
        with self._lock:
            # Impose token debt: forces all callers to wait until the penalty is refilled at normal rate
            self.tokens = -penalty_seconds * self.rate
            self.last_update = time.monotonic()
            if logfire:
                logfire.warn(
                    "Rate limiter throttled: {penalty_seconds:.2f}s penalty applied (RPM: {rpm})",
                    penalty_seconds=penalty_seconds,
                    rpm=self.rpm,
                )


@observe()
def get_rate_limiter(rpm: Optional[int] = None) -> TokenBucketRateLimiter:
    """Retrieve or instantiate a cached TokenBucketRateLimiter for the given RPM.

    What:
        Looks up an existing rate limiter keyed by target RPM in a global registry,
        or creates and caches a new instance under a thread-safe lock.

    Why:
        Rate limiting must be globally coordinated across all worker threads and tasks.
        Instantiating multiple independent limiters would split quota tracking and allow
        threads to overwhelm the API. A centralized singleton per RPM ensures unified pacing.
    """
    target_rpm = rpm or getattr(
        settings, "EMBEDDING_RPM_LIMIT", DEFAULT_RPM_LIMIT) or DEFAULT_RPM_LIMIT
    with _RATE_LIMITER_LOCK:
        if target_rpm not in _RATE_LIMITERS:
            _RATE_LIMITERS[target_rpm] = TokenBucketRateLimiter(rpm=target_rpm)
        return _RATE_LIMITERS[target_rpm]


def _extract_retry_delay(exc: Exception, default_delay: float) -> float:
    """Extract recommended retry delay in seconds from 429 error messages or headers.

    What:
        Inspects exception strings with regular expressions to find server-specified
        cooldown periods (e.g. 'retry-after: 5', 'wait 3.2s') and adds random jitter.

    Why:
        Upstream providers frequently communicate exact backoff requirements in 429 responses.
        Adhering to their requested duration prevents retrying too early (which prolongs bans),
        while adding uniform jitter (0.2s - 0.8s) prevents the 'thundering herd' problem where
        all blocked workers wake up and retry at the exact same millisecond.
    """
    msg = str(exc).lower()
    # Parse varied upstream delay strings: "retry-after: X", "seconds: X", or "wait Xs"
    match = re.search(r"retry[- ]after[:\s]+([0-9.]+)", msg)
    if not match:
        match = re.search(r"seconds[:\s]+([0-9.]+)", msg)
    if not match:
        match = re.search(r"wait\s+([0-9.]+)\s*s", msg)

    if match:
        try:
            # Add random jitter (0.2-0.8s) to mitigate thundering herd when multiple retries wake up
            return float(match.group(1)) + random.uniform(0.2, 0.8)
        except (ValueError, TypeError):
            pass
    return default_delay


# ==============================================================================
# CLIENT MANAGEMENT
# ==============================================================================


@observe()
def get_genai_client(api_key: Optional[str] = None) -> genai.Client:
    """Get or initialize a cached Google GenAI Client instance.

    What:
        Returns a configured genai.Client instance, caching it globally so subsequent
        calls reuse the same instance unless an explicit override API key is provided.

    Why:
        Creating a new Google GenAI Client per batch incurs significant network overhead
        (TLS handshakes, HTTP/2 connection negotiation, gRPC channel setup) adding 50-150ms
        of latency per batch. Reusing a persistent client enables connection pooling and
        keep-alive reuse, dramatically reducing overall pipeline runtime.
    """
    global _GENAI_CLIENT
    key = api_key or settings.GEMINI_API_KEY
    if not key:
        raise ValueError(
            "GEMINI_API_KEY is not set. Please provide it in your environment or config."
        )

    # Reusing the client keeps the internal HTTP/2 connection pool alive across requests
    if _GENAI_CLIENT is None or api_key is not None:
        client = genai.Client(api_key=key)
        if api_key is None:
            _GENAI_CLIENT = client
        return client
    return _GENAI_CLIENT


def _get_jina_api_key(api_key: Optional[str] = None) -> Optional[str]:
    """Retrieve Jina AI API key from explicit argument, settings, or environment.

    What:
        Inspects credentials hierarchically: explicit argument -> application settings
        -> JINA_API_KEY environment variable -> JINA_AI_API_TOKEN environment variable.

    Why:
        Centralizes key resolution logic following 12-factor application standards, allowing
        seamless fallback and flexible runtime overrides in containerized environments.
    """
    return (
        api_key
        or getattr(settings, "JINA_API_KEY", None)
        or os.getenv("JINA_API_KEY")
        or os.getenv("JINA_AI_API_TOKEN")
    )


@observe()
def get_jina_client(timeout: float = DEFAULT_TIMEOUT_SECONDS) -> httpx.Client:
    """Get or initialize a cached synchronous httpx.Client for Jina AI API calls.

    What:
        Returns a persistent httpx.Client configured with standard timeouts, recreating
        the client if the previous instance was closed.

    Why:
        Reusing an HTTP client maintains persistent TCP and TLS connections to api.jina.ai,
        minimizing latency overhead when executing fallback requests.
    """
    global _JINA_CLIENT
    if httpx is None:
        raise ImportError(
            "httpx is required for Jina AI embeddings. Install it via 'pip install httpx'."
        )
    # Recreate client if previous instance was closed
    if _JINA_CLIENT is None or _JINA_CLIENT.is_closed:
        _JINA_CLIENT = httpx.Client(timeout=timeout)
    return _JINA_CLIENT


@observe()
def get_async_jina_client(timeout: float = DEFAULT_TIMEOUT_SECONDS) -> httpx.AsyncClient:
    """Get or initialize a cached asynchronous httpx.AsyncClient for Jina AI API calls.

    What:
        Returns a persistent httpx.AsyncClient configured with standard timeouts,
        recreating it if the previous instance was closed.

    Why:
        Asynchronous operations in event-loop environments require non-blocking HTTP clients.
        Caching the async client provides connection pooling across async fallback tasks
        without blocking the event loop.
    """
    global _ASYNC_JINA_CLIENT
    if httpx is None:
        raise ImportError(
            "httpx is required for Jina AI embeddings. Install it via 'pip install httpx'."
        )
    # Recreate async client if previous instance was closed
    if _ASYNC_JINA_CLIENT is None or _ASYNC_JINA_CLIENT.is_closed:
        _ASYNC_JINA_CLIENT = httpx.AsyncClient(timeout=timeout)
    return _ASYNC_JINA_CLIENT


# ==============================================================================
# VECTOR NORMALIZATION & IMAGE UTILITIES
# ==============================================================================


@observe()
def normalize_vector(vector: Sequence[float]) -> List[float]:
    """L2-normalizes an embedding vector for exact cosine similarity search.

    What:
        Computes the Euclidean (L2) norm of the vector and divides each coordinate
        by the norm, returning a unit vector where ||v||_2 = 1.0. If the vector has
        zero magnitude or contains NaN values, returns the vector unchanged.

    Why:
        When embedding vectors are normalized to unit length, cosine similarity
        simplifies mathematically to the inner product (dot product):
        cos(u, v) = (u . v) / (||u|| * ||v||) = u . v.
        Vector databases (Chroma, FAISS, PGVector, Qdrant) execute dot-product
        SIMD operations significantly faster than full cosine distance calculations,
        yielding higher search throughput without sacrificing retrieval accuracy.
    """
    norm = math.sqrt(sum(x * x for x in vector))
    # Safeguard against zero or NaN vectors to prevent division by zero
    if norm == 0.0 or math.isnan(norm):
        return list(vector)
    # Unit vectors allow fast inner-product (dot product) search equivalent to cosine similarity
    return [x / norm for x in vector]


def _chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Partition a list into contiguous sublists of size at most chunk_size.

    What:
        Takes an input sequence and segments it into uniform batches of length
        chunk_size (with the final sublist containing any remaining items).

    Why:
        Embedding APIs reject requests that exceed maximum batch limits (e.g. Google's
        limit of 100 texts per call). Chunking prevents payload rejections while creating
        manageable work units that can be dispatched across parallel worker threads.
    """
    return [lst[i: i + chunk_size] for i in range(0, len(lst), chunk_size)]


def _sanitize_text(text: str) -> str:
    """Sanitize input text to ensure non-empty content for embedding models.

    What:
        Converts non-string inputs to strings, strips leading and trailing whitespace,
        and substitutes an empty or whitespace-only string with a single space (" ").

    Why:
        Both Google Gemini and Jina AI reject empty strings with HTTP 400 Bad Request
        errors. Filtering out or dropping empty texts would alter the length of the batch,
        breaking the 1-to-1 index correspondence between input document chunks and their
        resultant embedding vectors. Substituting a single space (" ") avoids API errors
        while preserving list length and array alignment.
    """
    if not isinstance(text, str):
        text = str(text)
    stripped = text.strip()
    # Embedding endpoints reject empty strings with 400 errors; space preserves batch index alignment
    return stripped if stripped else " "


def _to_image_bytes(item: ImageInput) -> tuple[bytes, str]:
    """Convert an arbitrary image representation to raw PNG bytes and a MIME type string.

    What:
        Inspects the input type and extracts raw binary image data along with its MIME type.
        Supports duck-typed ExtractedImage objects, PIL Image.Image instances, raw byte arrays,
        filesystem paths (str or Path), and Base64 Data URI strings.

    Why:
        Document loaders across the RAG ingestion pipeline produce varied image structures
        (e.g., PIL images from OCR, disk file paths from document parsers, data URIs from
        markdown/HTML extractors). Normalizing all variations to raw bytes and MIME types
        decouples downstream embedding logic from upstream extractor implementations.
    """
    # 1. Duck-typed ExtractedImage instance (from app.ingestion.loader)
    if hasattr(item, "image") and getattr(item, "image") is not None:
        img = item.image
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue(), "image/png"

    # 2. PIL Image.Image instance
    if isinstance(item, Image.Image):
        buf = io.BytesIO()
        item.save(buf, format="PNG")
        return buf.getvalue(), "image/png"

    # 3. Raw bytes
    if isinstance(item, (bytes, bytearray)):
        return bytes(item), "image/png"

    # 4. File path or Base64 data URI string
    if isinstance(item, (str, Path)):
        p = Path(item)
        if p.exists() and p.is_file():
            mime, _ = mimetypes.guess_type(p)
            return p.read_bytes(), mime or "image/png"
        elif isinstance(item, str) and item.startswith("data:image/"):
            # Strip standard RFC 2397 scheme ('data:<mime>;base64,<payload>') to decode raw bytes
            header, b64_data = item.split(",", 1)
            mime = header.split(";")[0].replace("data:", "")
            return base64.b64decode(b64_data), mime or "image/png"

    raise ValueError(
        f"Unsupported image input type for embedding: {type(item)}")


def _to_image_data_uri(item: ImageInput) -> str:
    """Convert an arbitrary image input to an RFC 2397 Base64 Data URI string.

    What:
        Converts image input to binary bytes via _to_image_bytes(), Base64-encodes
        the data, and formats it as 'data:<mime_type>;base64,<base64_encoded_string>'.
        Passes existing Data URI strings through unchanged.

    Why:
        Jina AI's multimodal REST API (/v1/embeddings) expects images formatted as Base64
        Data URIs inside the JSON payload ('inputs': [{'image': 'data:image/png;base64,...'}])
        rather than multipart/form-data attachments.
    """
    if isinstance(item, str) and item.startswith("data:image/"):
        return item
    img_bytes, mime = _to_image_bytes(item)
    b64_str = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64_str}"


def _map_task_to_jina(task_type: Optional[str]) -> str:
    """Map Google or generic retrieval task types to Jina AI task adapter heads.

    What:
        Translates task strings (e.g. 'RETRIEVAL_DOCUMENT', 'RETRIEVAL_QUERY', 'SEMANTIC_SIMILARITY')
        into Jina v4 task adapter identifiers: 'retrieval.passage', 'retrieval.query',
        or 'text-matching'.

    Why:
        Asymmetric retrieval models use dual-encoder architectures with specialized projection
        adapters for search queries versus document passages. Mapping Google's task types
        to Jina's equivalent adapters ensures that when failover occurs, queries and passages
        continue to project into mutually aligned vector spaces.
    """
    if not task_type:
        return "retrieval.passage"
    t = task_type.lower()
    # Asymmetric retrieval: queries and passages project into distinct complementary subspaces
    if "query" in t:
        return "retrieval.query"
    if "document" in t or "passage" in t:
        return "retrieval.passage"
    if "match" in t or "similarity" in t:
        return "text-matching"
    return "retrieval.passage"


# ==============================================================================
# SINGLE-BATCH TEXT EMBEDDING WITH RATE LIMITING
# ==============================================================================


@observe()
def _embed_google_batch_with_retry(
    client: genai.Client,
    batch: List[str],
    model: str,
    dimension: Optional[int],
    task_type: Optional[str],
    max_retries: int,
    timeout: float,
    rate_limiter: TokenBucketRateLimiter,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> List[List[float]]:
    """Execute synchronous batch text embedding against Google GenAI with rate limiting & retry.

    What:
        Dispatches a batch slice of text strings to Google's embed_content API with the desired
        output dimensionality and retrieval task type. For Gemini Embedding 2 ('gemini-embedding-2'),
        concurrently embeds individual items across a thread pool. Implements exponential backoff
        with jitter and coordinates rate-limit penalties across threads.

    Why:
        1. 1-to-1 Vector Generation: Unlike legacy embedding models (e.g. text-embedding-004),
           gemini-embedding-2 interprets a list of contents as multimodal parts of a SINGLE composite
           document, fusing them into ONE aggregated embedding. Embedding items individually in
           parallel threads is required to generate distinct 1:1 embeddings for each document chunk.
        2. Quota Management: Rate-limiter token acquisition prevents client-side flooding of Google's
           RPM limit, while throttle() imposes a token deficit upon encountering 429 to pause all
           concurrent worker threads simultaneously.
        3. Fast Failure: Client errors (HTTP 400 Bad Request, 404 Model Not Found) fail immediately
           without consuming retry attempts, while transient network and quota errors back off safely.
    """
    config_kwargs: dict[str, Any] = {
        "http_options": types.HttpOptions(timeout=int(timeout * 1000))
    }
    if dimension is not None:
        config_kwargs["output_dimensionality"] = dimension
    if task_type:
        config_kwargs["task_type"] = task_type

    config = types.EmbedContentConfig(**config_kwargs)
    is_gemini_2 = "gemini-embedding-2" in model.lower()

    with _logfire_span(
        "google_genai.embed_batch",
        model=model,
        batch_size=len(batch),
        dimension=dimension,
        task_type=task_type,
        is_gemini_2=is_gemini_2,
    ):
        # Strategic: Gemini Embedding 2 merges multi-item contents into a single aggregated vector.
        # To obtain individual 1-to-1 embeddings for chunks, dispatch them concurrently.
        if is_gemini_2 and len(batch) > 1:
            def _embed_single(text: str) -> List[float]:
                last_err = None
                for attempt in range(1, max_retries + 1):
                    # Throttle against shared token bucket
                    rate_limiter.acquire(1)
                    try:
                        res = client.models.embed_content(
                            model=model,
                            contents=text,
                            config=config,
                        )
                        return list(res.embeddings[0].values)
                    except Exception as e:
                        last_err = e
                        err_str = str(e).lower()
                        # Fail fast on non-transient errors (bad model name, invalid request params)
                        if "404" in err_str or "not found" in err_str or "400" in err_str:
                            raise e

                        if attempt < max_retries:
                            backoff = (2 ** (attempt - 1)) + \
                                random.uniform(0.2, 0.6)
                            # Reactive 429 backoff: pause the shared limiter to hold other threads
                            if "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str:
                                backoff = _extract_retry_delay(
                                    e, default_delay=backoff * 2.0)
                                rate_limiter.throttle(backoff)
                                logger.warning(
                                    f"Rate limit (429) hit for '{model}'. Throttling rate limiter and "
                                    f"retrying in {backoff:.2f}s (attempt {attempt}/{max_retries})..."
                                )
                            else:
                                logger.warning(
                                    f"Google embedding attempt {attempt}/{max_retries} failed for '{model}': {e}. "
                                    f"Retrying in {backoff:.2f}s..."
                                )
                            time.sleep(backoff)
                        else:
                            raise e
                raise last_err or RuntimeError("Single item embedding failed")

            workers = min(max_concurrency, len(batch))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                return list(executor.map(_embed_single, batch))

        # Single item for Gemini 2, or standard list batching for legacy models (e.g. text-embedding-004)
        contents_input = batch[0] if (
            is_gemini_2 and len(batch) == 1) else batch

        last_error: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            rate_limiter.acquire(1)
            try:
                res = client.models.embed_content(
                    model=model,
                    contents=contents_input,
                    config=config,
                )
                embeddings = [list(e.values) for e in res.embeddings]
                return embeddings

            except (ClientError, APIError, Exception) as e:
                last_error = e
                err_str = str(e).lower()
                if "404" in err_str or "not found" in err_str or "400" in err_str:
                    logger.warning(
                        f"Google embedding model '{model}' non-retryable error: {e}")
                    raise e

                if attempt < max_retries:
                    backoff = (2 ** (attempt - 1)) + random.uniform(0.2, 0.6)
                    if "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str:
                        backoff = _extract_retry_delay(
                            e, default_delay=backoff * 2.0)
                        rate_limiter.throttle(backoff)
                        logger.warning(
                            f"Rate limit (429) hit on model '{model}'. Throttling and retrying in {backoff:.2f}s "
                            f"(attempt {attempt}/{max_retries})..."
                        )
                    else:
                        logger.warning(
                            f"Google embedding attempt {attempt}/{max_retries} failed for model '{model}': {e}. "
                            f"Retrying in {backoff:.2f}s..."
                        )
                    time.sleep(backoff)
                else:
                    logger.error(
                        f"Google embedding all {max_retries} attempts failed for model '{model}': {e}"
                    )
                    raise last_error

        raise last_error or RuntimeError("Unexpected embedding failure")


@observe()
async def _aembed_google_batch_with_retry(
    client: genai.Client,
    batch: List[str],
    model: str,
    dimension: Optional[int],
    task_type: Optional[str],
    max_retries: int,
    timeout: float,
    rate_limiter: TokenBucketRateLimiter,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> List[List[float]]:
    """Execute asynchronous batch text embedding against Google GenAI with rate limiting & retry.

    What:
        Asynchronous equivalent to _embed_google_batch_with_retry. Uses client.aio.models.embed_content
        with an asyncio.Semaphore to bound concurrent in-flight requests. Concurrently embeds individual
        chunks for Gemini 2 models, applying non-blocking asyncio.sleep backoff on errors.

    Why:
        Non-blocking async I/O is critical for event-driven server architectures (FastAPI, asyncio)
        to handle hundreds of concurrent requests without thread starvation. The semaphore bounds
        in-flight network sockets to prevent socket exhaustion, while async token acquisition ensures
        the event loop remains responsive during rate-limit pauses.
    """
    config_kwargs: dict[str, Any] = {
        "http_options": types.HttpOptions(timeout=int(timeout * 1000))
    }
    if dimension is not None:
        config_kwargs["output_dimensionality"] = dimension
    if task_type:
        config_kwargs["task_type"] = task_type

    config = types.EmbedContentConfig(**config_kwargs)
    is_gemini_2 = "gemini-embedding-2" in model.lower()

    with _logfire_span(
        "google_genai.aembed_batch",
        model=model,
        batch_size=len(batch),
        dimension=dimension,
        task_type=task_type,
        is_gemini_2=is_gemini_2,
    ):
        # In Gemini Embedding 2, embed individual items asynchronously to avoid vector aggregation
        if is_gemini_2 and len(batch) > 1:
            # Limit concurrent async tasks to bounded concurrency
            sem = asyncio.Semaphore(max_concurrency)

            async def _aembed_single(text: str) -> List[float]:
                async with sem:
                    last_err = None
                    for attempt in range(1, max_retries + 1):
                        await rate_limiter.acquire_async(1)
                        try:
                            res = await client.aio.models.embed_content(
                                model=model,
                                contents=text,
                                config=config,
                            )
                            return list(res.embeddings[0].values)
                        except Exception as e:
                            last_err = e
                            err_str = str(e).lower()
                            if "404" in err_str or "not found" in err_str or "400" in err_str:
                                raise e

                            if attempt < max_retries:
                                backoff = (2 ** (attempt - 1)) + \
                                    random.uniform(0.2, 0.6)
                                if "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str:
                                    backoff = _extract_retry_delay(
                                        e, default_delay=backoff * 2.0)
                                    rate_limiter.throttle(backoff)
                                    logger.warning(
                                        f"Async rate limit (429) hit for '{model}'. Throttling and retrying in {backoff:.2f}s..."
                                    )
                                await asyncio.sleep(backoff)
                            else:
                                raise e
                    raise last_err or RuntimeError(
                        "Single item async embedding failed")

            tasks = [_aembed_single(t) for t in batch]
            return await asyncio.gather(*tasks)

        # Standard batching path for single items or models supporting direct list batching
        contents_input = batch[0] if (
            is_gemini_2 and len(batch) == 1) else batch

        last_error: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            await rate_limiter.acquire_async(1)
            try:
                res = await client.aio.models.embed_content(
                    model=model,
                    contents=contents_input,
                    config=config,
                )
                embeddings = [list(e.values) for e in res.embeddings]
                return embeddings

            except (ClientError, APIError, Exception) as e:
                last_error = e
                err_str = str(e).lower()
                if "404" in err_str or "not found" in err_str or "400" in err_str:
                    logger.warning(
                        f"Google embedding model '{model}' non-retryable error: {e}")
                    raise e

                if attempt < max_retries:
                    backoff = (2 ** (attempt - 1)) + random.uniform(0.2, 0.6)
                    if "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str:
                        backoff = _extract_retry_delay(
                            e, default_delay=backoff * 2.0)
                        rate_limiter.throttle(backoff)
                        logger.warning(
                            f"Async rate limit (429) hit on '{model}'. Throttling and retrying in {backoff:.2f}s..."
                        )
                    else:
                        logger.warning(
                            f"Async Google embedding attempt {attempt}/{max_retries} failed for '{model}': {e}. "
                            f"Retrying in {backoff:.2f}s..."
                        )
                    await asyncio.sleep(backoff)
                else:
                    logger.error(
                        f"Async Google embedding all {max_retries} attempts failed for '{model}': {e}"
                    )
                    raise last_error

        raise last_error or RuntimeError("Unexpected async embedding failure")


# ==============================================================================
# JINA AI MULTIMODAL EMBEDDING HANDLERS (PRIMARY FALLBACK)
# ==============================================================================


@observe()
def _embed_jina_batch(
    inputs: List[Union[str, Dict[str, Any]]],
    model: str = DEFAULT_FALLBACK_MODEL,
    dimension: Optional[int] = DEFAULT_DIMENSION,
    task_type: Optional[str] = DOCUMENT_TASK_TYPE,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    api_key: Optional[str] = None,
) -> List[List[float]]:
    """Embed a batch of text or multimodal inputs using Jina AI API (jina-embeddings-v4).

    What:
        Submits text strings or multimodal dicts (e.g. {'image': 'data:image/...'}) to Jina AI's
        /v1/embeddings endpoint. Configures task adapter heads, enforces Matryoshka Representation
        Learning (MRL) dimension truncation, and re-orders returned vectors by item index.

    Why:
        1. Multimodal Fallback: Serves as the primary failover provider when Google Gemini quota
           is exhausted or encounters network downtime.
        2. Schema Consistency (MRL): Jina v4 outputs 2048-dimensional vectors by default. Passing
           dimensions=768 uses MRL truncation to match Gemini's dimensionality, preventing vector
           database schema mismatches.
        3. Deterministic Order: Jina returns an 'index' field with each embedding item. Sorting by
           index guarantees that output vectors match the input sequence order even if processed
           out of order by Jina's distributed inference backend.
    """
    key = _get_jina_api_key(api_key)
    if not key:
        raise ValueError(
            "JINA_API_KEY is not set. Please set JINA_API_KEY in your .env or pass api_key."
        )

    client = get_jina_client(timeout=timeout)
    jina_task = _map_task_to_jina(task_type)

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    payload: Dict[str, Any] = {
        "model": model,
        "task": jina_task,
        "input": inputs,
    }
    # Matryoshka Representation Learning (MRL): truncates dimensions to match primary Gemini vectors
    if dimension is not None:
        payload["dimensions"] = dimension

    with _logfire_span(
        "jina_ai.embed_batch",
        model=model,
        batch_size=len(inputs),
        task=jina_task,
        dimension=dimension,
    ):
        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = client.post(JINA_API_URL, headers=headers, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    items = data.get("data", [])
                    # Re-sort by input index to preserve strict 1:1 order alignment with caller inputs
                    sorted_items = sorted(
                        items, key=lambda x: x.get("index", 0))
                    return [item["embedding"] for item in sorted_items]

                status = resp.status_code
                if status == 429:
                    # Respect upstream 'Retry-After' header if present to avoid immediate re-trigger
                    retry_after = float(resp.headers.get("Retry-After", 2.0))
                    backoff = max(retry_after, (2 ** (attempt - 1)
                                                ) + random.uniform(0.2, 0.8))
                    logger.warning(
                        f"Jina AI rate limit (429) hit for '{model}'. Retrying in {backoff:.2f}s "
                        f"(attempt {attempt}/{max_retries})..."
                    )
                    time.sleep(backoff)
                    continue
                elif 500 <= status < 600:
                    backoff = (2 ** (attempt - 1)) + random.uniform(0.2, 0.8)
                    logger.warning(
                        f"Jina AI server error ({status}) for '{model}'. Retrying in {backoff:.2f}s "
                        f"(attempt {attempt}/{max_retries})..."
                    )
                    time.sleep(backoff)
                    continue
                else:
                    resp.raise_for_status()

            except Exception as e:
                last_exc = e
                err_str = str(e).lower()
                if attempt < max_retries and ("429" in err_str or "50" in err_str or "timeout" in err_str):
                    backoff = (2 ** (attempt - 1)) + random.uniform(0.2, 0.8)
                    time.sleep(backoff)
                else:
                    raise e

        raise last_exc or RuntimeError(
            f"Failed to embed batch with Jina AI '{model}'")


@observe()
async def _aembed_jina_batch(
    inputs: List[Union[str, Dict[str, Any]]],
    model: str = DEFAULT_FALLBACK_MODEL,
    dimension: Optional[int] = DEFAULT_DIMENSION,
    task_type: Optional[str] = DOCUMENT_TASK_TYPE,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    api_key: Optional[str] = None,
) -> List[List[float]]:
    """Asynchronously embed a batch of inputs using Jina AI API (jina-embeddings-v4).

    What:
        Asynchronous counterpart to _embed_jina_batch. Uses httpx.AsyncClient to dispatch
        text or multimodal payloads to Jina AI's /v1/embeddings endpoint.

    Why:
        Provides non-blocking fallback capability for asynchronous execution paths,
        ensuring that transient Google failures do not block the asyncio event loop.
    """
    key = _get_jina_api_key(api_key)
    if not key:
        raise ValueError(
            "JINA_API_KEY is not set. Please set JINA_API_KEY in your .env or pass api_key."
        )

    client = get_async_jina_client(timeout=timeout)
    jina_task = _map_task_to_jina(task_type)

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    payload: Dict[str, Any] = {
        "model": model,
        "task": jina_task,
        "input": inputs,
    }
    # Matryoshka Representation Learning (MRL): enforce matching dimensionality asynchronously
    if dimension is not None:
        payload["dimensions"] = dimension

    with _logfire_span(
        "jina_ai.aembed_batch",
        model=model,
        batch_size=len(inputs),
        task=jina_task,
        dimension=dimension,
    ):
        last_exc = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = await client.post(JINA_API_URL, headers=headers, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    items = data.get("data", [])
                    # Re-sort items by input index to maintain deterministic sequence order
                    sorted_items = sorted(
                        items, key=lambda x: x.get("index", 0))
                    return [item["embedding"] for item in sorted_items]

                status = resp.status_code
                if status == 429:
                    retry_after = float(resp.headers.get("Retry-After", 2.0))
                    backoff = max(retry_after, (2 ** (attempt - 1)
                                                ) + random.uniform(0.2, 0.8))
                    logger.warning(
                        f"Async Jina AI rate limit (429) hit for '{model}'. Retrying in {backoff:.2f}s "
                        f"(attempt {attempt}/{max_retries})..."
                    )
                    await asyncio.sleep(backoff)
                    continue
                elif 500 <= status < 600:
                    backoff = (2 ** (attempt - 1)) + random.uniform(0.2, 0.8)
                    logger.warning(
                        f"Async Jina AI server error ({status}) for '{model}'. Retrying in {backoff:.2f}s..."
                    )
                    await asyncio.sleep(backoff)
                    continue
                else:
                    resp.raise_for_status()

            except Exception as e:
                last_exc = e
                err_str = str(e).lower()
                if attempt < max_retries and ("429" in err_str or "50" in err_str or "timeout" in err_str):
                    backoff = (2 ** (attempt - 1)) + random.uniform(0.2, 0.8)
                    await asyncio.sleep(backoff)
                else:
                    raise e

        raise last_exc or RuntimeError(
            f"Async embedding failed with Jina AI '{model}'")


# ==============================================================================
# FALLBACK EMBEDDING ROUTERS (JINA EMBEDDINGS V4)
# ==============================================================================


@observe()
def _embed_fallback_batch(
    batch: List[str],
    fallback_model: str = DEFAULT_FALLBACK_MODEL,
    dimension: Optional[int] = DEFAULT_DIMENSION,
    task_type: Optional[str] = DOCUMENT_TASK_TYPE,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    api_key: Optional[str] = None,
) -> List[List[float]]:
    """Route batch text embedding to the fallback model (jina-embeddings-v4).

    What:
        Wrapper that prepares and routes a slice of text strings to _embed_jina_batch.

    Why:
        Isolates text fallback routing logic from higher-level batch orchestration,
        providing a single cleanly typed point of contact for text failover.
    """
    return _embed_jina_batch(
        inputs=batch,
        model=fallback_model,
        dimension=dimension,
        task_type=task_type,
        timeout=timeout,
        max_retries=max_retries,
        api_key=api_key,
    )


@observe()
async def _aembed_fallback_batch(
    batch: List[str],
    fallback_model: str = DEFAULT_FALLBACK_MODEL,
    dimension: Optional[int] = DEFAULT_DIMENSION,
    task_type: Optional[str] = DOCUMENT_TASK_TYPE,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    api_key: Optional[str] = None,
) -> List[List[float]]:
    """Asynchronously route batch text embedding to the fallback model (jina-embeddings-v4).

    What:
        Asynchronous wrapper that routes a slice of text strings to _aembed_jina_batch.

    Why:
        Enables non-blocking asynchronous failover routing for text batching pipelines.
    """
    return await _aembed_jina_batch(
        inputs=batch,
        model=fallback_model,
        dimension=dimension,
        task_type=task_type,
        timeout=timeout,
        max_retries=max_retries,
        api_key=api_key,
    )


@observe()
def _embed_image_fallback(
    image: ImageInput,
    fallback_model: str = DEFAULT_FALLBACK_MODEL,
    dimension: Optional[int] = DEFAULT_DIMENSION,
    task_type: Optional[str] = DOCUMENT_TASK_TYPE,
    api_key: Optional[str] = None,
) -> List[float]:
    """Route image embedding to the fallback model (jina-embeddings-v4) using native image input.

    What:
        Converts the image input to an RFC 2397 Data URI and passes it to Jina AI's multimodal API,
        returning the first embedding vector.

    Why:
        Preserves multimodal retrieval capabilities when the primary Google GenAI image model
        is unavailable or rate-limited, embedding the image into the same 768-dimensional space.
    """
    # Convert image to RFC 2397 data URI string format required by Jina's multimodal endpoint
    with _logfire_span("jina_ai.embed_image_fallback", model=fallback_model, dimension=dimension, task_type=task_type):
        data_uri = _to_image_data_uri(image)
        embs = _embed_jina_batch(
            inputs=[{"image": data_uri}],
            model=fallback_model,
            dimension=dimension,
            task_type=task_type,
            api_key=api_key,
        )
        return embs[0]


@observe()
async def _aembed_image_fallback(
    image: ImageInput,
    fallback_model: str = DEFAULT_FALLBACK_MODEL,
    dimension: Optional[int] = DEFAULT_DIMENSION,
    task_type: Optional[str] = DOCUMENT_TASK_TYPE,
    api_key: Optional[str] = None,
) -> List[float]:
    """Asynchronously route image embedding to the fallback model (jina-embeddings-v4) using native image input.

    What:
        Converts the image input to a Data URI and asynchronously requests an embedding vector
        from Jina AI's multimodal endpoint via _aembed_jina_batch.

    Why:
        Provides non-blocking multimodal fallback for asynchronous image processing pipelines.
    """
    with _logfire_span("jina_ai.aembed_image_fallback", model=fallback_model, dimension=dimension, task_type=task_type):
        data_uri = _to_image_data_uri(image)
        embs = await _aembed_jina_batch(
            inputs=[{"image": data_uri}],
            model=fallback_model,
            dimension=dimension,
            task_type=task_type,
            api_key=api_key,
        )
        return embs[0]


# ==============================================================================
# MAIN TEXT EMBEDDING FUNCTIONS
# ==============================================================================


@observe()
def embed_texts(
    texts: Union[str, List[str]],
    model: str = DEFAULT_PRIMARY_MODEL,
    dimension: Optional[int] = DEFAULT_DIMENSION,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    task_type: str = DOCUMENT_TASK_TYPE,
    fallback_model: Optional[str] = DEFAULT_FALLBACK_MODEL,
    rpm_limit: Optional[int] = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    normalize: bool = True,
    api_key: Optional[str] = None,
    prefer_fallback: bool = False,
) -> List[List[float]]:
    """Embed a collection of texts with rate limiting, batching, and Jina AI fallback.

    What:
        Sanitizes input texts, partitions them into slices of size up to MAX_GOOGLE_BATCH_SIZE (100),
        and concurrently embeds them across a thread pool using Google Gemini Embedding 2.
        If the primary model fails on any batch slice after exhausting retries, that specific slice
        is transparently failed over to Jina Embeddings v4. Results are re-assembled into exact
        original input order and optionally L2-normalized.

    Why:
        1. Throughput & Scalability: Slicing into bounded batches combined with ThreadPoolExecutor
           concurrency optimizes HTTP/gRPC bandwidth and GPU utilization without violating API limits.
        2. Per-Batch Fault Isolation: Failing over at the batch slice level ensures that transient
           Google quota or server errors on a single slice do not invalidate earlier or later successful
           batches, making large ingestion runs resilient.
        3. Deterministic Sequence Ordering: Pre-allocated indexed result slots prevent race conditions
           and preserve exact 1-to-1 chunk-to-embedding alignment even as threads finish out-of-order.
        4. Vector Search Optimization: L2-normalizing vectors enables high-performance inner product
           (dot product) searches in vector databases (Chroma, FAISS, PGVector).

    Args:
        texts: A single text string or a list of text strings/chunks to embed.
        model: Primary Google embedding model (default: 'gemini-embedding-2').
        dimension: Desired embedding vector dimension (default: 768).
        batch_size: Number of texts per batch slice (default: 32).
        max_concurrency: Number of concurrent batch worker threads (default: 8).
        task_type: Retrieval task type (default: 'RETRIEVAL_DOCUMENT').
        fallback_model: Fallback model if primary fails (default: 'jina-embeddings-v4').
        rpm_limit: Requests-per-minute throttle (default: 600 or from settings).
        timeout: Request timeout in seconds per batch (default: 30.0).
        max_retries: Max retry attempts per batch on transient errors (default: 3).
        normalize: Whether to L2-normalize vectors for cosine similarity (default: True).
        api_key: Optional Gemini API key override.
        prefer_fallback: Whether to directly use the fallback/fallout model.

    Returns:
        List[List[float]]: List of embedding vectors in strict 1-to-1 correspondence with input texts.
    """
    if isinstance(texts, str):
        raw_list = [texts]
    else:
        raw_list = list(texts)

    if not raw_list:
        return []

    sanitized_texts = [_sanitize_text(t) for t in raw_list]
    # Cap batch slice size at Google's maximum payload limit (100) to prevent payload rejections
    effective_batch_size = min(max(1, batch_size), MAX_GOOGLE_BATCH_SIZE)

    client = get_genai_client(api_key=api_key)
    rate_limiter = get_rate_limiter(rpm=rpm_limit)
    batches = _chunk_list(sanitized_texts, effective_batch_size)
    num_batches = len(batches)

    logger.debug(
        f"Embedding {len(sanitized_texts)} texts into {num_batches} batches "
        f"(batch_size={effective_batch_size}, dimension={dimension}, model='{model}', fallback='{fallback_model}')"
    )

    with _logfire_span(
        "embed_texts",
        total_texts=len(sanitized_texts),
        batch_size=effective_batch_size,
        num_batches=num_batches,
        model=model,
        fallback_model=fallback_model,
        dimension=dimension,
        task_type=task_type,
        normalize=normalize,
    ) as root_span:
        def _process_batch(batch_idx: int, batch_texts: List[str]) -> tuple[int, List[List[float]]]:
            with _logfire_span(
                "embed_texts.batch",
                batch_index=batch_idx + 1,
                num_batches=num_batches,
                item_count=len(batch_texts),
                model=model,
            ):
                # If fallout/fallback is preferred, route directly to fallback model
                if prefer_fallback or getattr(settings, "USE_FALLOUT_MODEL", False):
                    fallback_embs = _embed_fallback_batch(
                        batch=batch_texts,
                        fallback_model=fallback_model or DEFAULT_FALLBACK_MODEL,
                        dimension=dimension,
                        task_type=task_type,
                        timeout=timeout,
                        api_key=api_key,
                    )
                    return batch_idx, fallback_embs

                # Attempts primary embedding; transparently fails over only this batch slice to Jina on error
                try:
                    embs = _embed_google_batch_with_retry(
                        client=client,
                        batch=batch_texts,
                        model=model,
                        dimension=dimension,
                        task_type=task_type,
                        max_retries=max_retries,
                        timeout=timeout,
                        rate_limiter=rate_limiter,
                        max_concurrency=max_concurrency,
                    )
                    return batch_idx, embs
                except Exception as primary_exc:
                    if fallback_model:
                        logger.warning(
                            f"Primary model '{model}' failed on batch {batch_idx + 1}/{num_batches}: {primary_exc}. "
                            f"Failing over to fallback model '{fallback_model}'..."
                        )
                        if logfire:
                            logfire.warn(
                                "Primary model {model} failed on batch {batch_idx}/{num_batches}, failing over to {fallback_model}: {error}",
                                model=model,
                                batch_idx=batch_idx + 1,
                                num_batches=num_batches,
                                fallback_model=fallback_model,
                                error=str(primary_exc),
                            )
                        try:
                            fallback_embs = _embed_fallback_batch(
                                batch=batch_texts,
                                fallback_model=fallback_model,
                                dimension=dimension,
                                task_type=task_type,
                                timeout=timeout,
                                api_key=api_key,
                            )
                            return batch_idx, fallback_embs
                        except Exception as fallback_exc:
                            logger.error(
                                f"Both primary model '{model}' and fallback model '{fallback_model}' failed "
                                f"on batch {batch_idx + 1}: {fallback_exc}"
                            )
                            raise fallback_exc from primary_exc
                    else:
                        raise primary_exc

        # Pre-allocated indexed slots allow worker threads to place results out-of-order safely without race conditions
        ordered_results: List[Optional[List[List[float]]]] = [
            None] * num_batches

        if num_batches == 1:
            _, ordered_results[0] = _process_batch(0, batches[0])
        else:
            concurrency = min(max_concurrency, num_batches)
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                future_to_idx = {
                    executor.submit(_process_batch, i, batch): i for i, batch in enumerate(batches)
                }
                # Record batches into their designated index as threads finish
                for future in as_completed(future_to_idx):
                    idx, embs = future.result()
                    ordered_results[idx] = embs

        flattened_embeddings: List[List[float]] = []
        for batch_embs in ordered_results:
            if batch_embs is not None:
                flattened_embeddings.extend(batch_embs)

        # Unit normalization enables fast inner product (dot product) vector similarity searches
        if normalize:
            flattened_embeddings = [normalize_vector(
                vec) for vec in flattened_embeddings]

        total_chars = sum(len(t) for t in sanitized_texts)
        est_tokens = sum(max(1, max(int(len(t.split()) * 1.3), len(t) // 4)) for t in sanitized_texts)

        if root_span:
            root_span.set_attribute("vector_count", len(flattened_embeddings))
            if flattened_embeddings:
                root_span.set_attribute(
                    "vector_dim", len(flattened_embeddings[0]))
            root_span.set_attribute("tokens.estimated_total", est_tokens)
            root_span.set_attribute("tokens.character_count", total_chars)
            root_span.set_attribute("tokens.avg_tokens_per_item", round(est_tokens / len(sanitized_texts), 2) if sanitized_texts else 0.0)
            root_span.set_attribute("embeddings.vector_count", len(flattened_embeddings))
            root_span.set_attribute("embeddings.dimension", len(flattened_embeddings[0]) if flattened_embeddings else 0)

        return flattened_embeddings


@observe()
def embed_text(
    text: str,
    model: str = DEFAULT_PRIMARY_MODEL,
    dimension: Optional[int] = DEFAULT_DIMENSION,
    task_type: str = DOCUMENT_TASK_TYPE,
    fallback_model: Optional[str] = DEFAULT_FALLBACK_MODEL,
    rpm_limit: Optional[int] = None,
    normalize: bool = True,
    api_key: Optional[str] = None,
    prefer_fallback: bool = False,
) -> List[float]:
    """Embed a single text string using Google Gemini Embedding 2 with Jina Embeddings v4 fallback.

    What:
        Synchronously embeds an individual text string into a single dense vector (default: 768 dimensions)
        by routing through embed_texts with batch_size=1 and max_concurrency=1.

    Why:
        Provides an ergonomic, direct API for single-chunk embedding without requiring callers to wrap
        the input in a list or extract the first result from a returned 2D array.
    """
    with _logfire_span(
        "embed_text",
        text_length=len(text) if text else 0,
        model=model,
        dimension=dimension,
        task_type=task_type,
    ):
        results = embed_texts(
            texts=[text],
            model=model,
            dimension=dimension,
            batch_size=1,
            max_concurrency=1,
            task_type=task_type,
            fallback_model=fallback_model,
            rpm_limit=rpm_limit,
            normalize=normalize,
            api_key=api_key,
            prefer_fallback=prefer_fallback,
        )
        return results[0] if results else []


@observe()
def embed_query(
    query: str,
    model: str = DEFAULT_PRIMARY_MODEL,
    dimension: Optional[int] = DEFAULT_DIMENSION,
    fallback_model: Optional[str] = DEFAULT_FALLBACK_MODEL,
    rpm_limit: Optional[int] = None,
    normalize: bool = True,
    api_key: Optional[str] = None,
    prefer_fallback: bool = False,
) -> List[float]:
    """Embed a search query using Google Gemini Embedding 2 with task_type='RETRIEVAL_QUERY'.

    What:
        Generates an embedding vector for user search queries, explicitly setting task_type to
        'RETRIEVAL_QUERY' and routing through embed_text.

    Why:
        Modern dual-encoder retrieval models employ asymmetric projection spaces: queries and
        document passages are embedded differently. Setting task_type='RETRIEVAL_QUERY' optimizes
        the query vector to match against document passages embedded with 'RETRIEVAL_DOCUMENT',
        significantly improving search recall and precision compared to symmetric embeddings.
    """
    with _logfire_span(
        "embed_query",
        query_length=len(query) if query else 0,
        model=model,
        dimension=dimension,
        fallback_model=fallback_model,
        prefer_fallback=prefer_fallback,
    ):
        return embed_text(
            text=query,
            model=model,
            dimension=dimension,
            task_type=QUERY_TASK_TYPE,
            fallback_model=fallback_model,
            rpm_limit=rpm_limit,
            normalize=normalize,
            api_key=api_key,
            prefer_fallback=prefer_fallback,
        )


# ==============================================================================
# MULTIMODAL & IMAGE EMBEDDING FUNCTIONS
# ==============================================================================


@observe()
def embed_image(
    image: ImageInput,
    model: str = DEFAULT_PRIMARY_MODEL,
    dimension: Optional[int] = DEFAULT_DIMENSION,
    task_type: str = DOCUMENT_TASK_TYPE,
    fallback_model: Optional[str] = DEFAULT_FALLBACK_MODEL,
    rpm_limit: Optional[int] = None,
    normalize: bool = True,
    api_key: Optional[str] = None,
    prefer_fallback: bool = False,
) -> List[float]:
    """Embed an image using Google Gemini Embedding 2 with Jina Embeddings v4 multimodal fallback."""
    with _logfire_span(
        "embed_image",
        model=model,
        dimension=dimension,
        task_type=task_type,
        fallback_model=fallback_model,
        prefer_fallback=prefer_fallback,
    ):
        if prefer_fallback or getattr(settings, "USE_FALLOUT_MODEL", False):
            vec = _embed_image_fallback(
                image=image,
                fallback_model=fallback_model or DEFAULT_FALLBACK_MODEL,
                dimension=dimension,
                task_type=task_type,
                api_key=api_key,
            )
            return normalize_vector(vec) if normalize else vec

        client = get_genai_client(api_key=api_key)
        rate_limiter = get_rate_limiter(rpm=rpm_limit)
        rate_limiter.acquire(1)

        img_bytes, mime = _to_image_bytes(image)

        config_kwargs: dict[str, Any] = {
            "http_options": types.HttpOptions(timeout=int(DEFAULT_TIMEOUT_SECONDS * 1000))
        }
        if dimension is not None:
            config_kwargs["output_dimensionality"] = dimension
        if task_type:
            config_kwargs["task_type"] = task_type
        config = types.EmbedContentConfig(**config_kwargs)

        try:
            # Wrap image bytes in a Gemini Part for native visual embedding
            res = client.models.embed_content(
                model=model,
                contents=[types.Part.from_bytes(
                    data=img_bytes, mime_type=mime)],
                config=config,
            )
            vec = list(res.embeddings[0].values)
        except Exception as primary_exc:
            logger.warning(
                f"Primary image embedding with '{model}' failed: {primary_exc}. "
                f"Failing over to fallback model '{fallback_model}'..."
            )
            if logfire:
                logfire.warn(
                    "Primary image embedding with {model} failed, failing over to {fallback_model}: {error}",
                    model=model,
                    fallback_model=fallback_model,
                    error=str(primary_exc),
                )
            try:
                vec = _embed_image_fallback(
                    image=image,
                    fallback_model=fallback_model or DEFAULT_FALLBACK_MODEL,
                    dimension=dimension,
                    task_type=task_type,
                    api_key=api_key,
                )
            except Exception as fallback_exc:
                logger.error(
                    f"Multimodal image fallback '{fallback_model}' also failed: {fallback_exc}")
                raise fallback_exc from primary_exc

        return normalize_vector(vec) if normalize else vec


@observe()
def embed_images(
    images: Sequence[ImageInput],
    model: str = DEFAULT_PRIMARY_MODEL,
    dimension: Optional[int] = DEFAULT_DIMENSION,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    task_type: str = DOCUMENT_TASK_TYPE,
    fallback_model: Optional[str] = DEFAULT_FALLBACK_MODEL,
    rpm_limit: Optional[int] = None,
    normalize: bool = True,
    api_key: Optional[str] = None,
    prefer_fallback: bool = False,
) -> List[List[float]]:
    """Concurrently embed a collection of images using Gemini Embedding 2 with Jina AI fallback."""
    if not images:
        return []

    with _logfire_span(
        "embed_images",
        image_count=len(images),
        model=model,
        dimension=dimension,
        max_concurrency=max_concurrency,
        prefer_fallback=prefer_fallback,
    ):
        workers = min(max_concurrency, len(images))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    embed_image,
                    img,
                    model=model,
                    dimension=dimension,
                    task_type=task_type,
                    fallback_model=fallback_model,
                    rpm_limit=rpm_limit,
                    normalize=normalize,
                    api_key=api_key,
                    prefer_fallback=prefer_fallback,
                )
                for img in images
            ]
            return [f.result() for f in futures]


@observe()
def embed_multimodal(
    items: Sequence[Union[str, ImageInput, Dict[str, Any]]],
    model: str = DEFAULT_PRIMARY_MODEL,
    dimension: Optional[int] = DEFAULT_DIMENSION,
    fallback_model: Optional[str] = DEFAULT_FALLBACK_MODEL,
    normalize: bool = True,
    api_key: Optional[str] = None,
) -> List[List[float]]:
    """Embed a heterogeneous collection of text, standalone images, or paired text+image items.

    What:
        Iterates over a mixed sequence of items, routing plain strings to embed_text, standalone
        images to embed_image, and paired dicts ({'text': ..., 'image': ...}) to a joint
        multi-part embed_content call combining text and image bytes into a single request.

    Why:
        Complex document layouts (e.g. PDF figures with accompanying captions, slides with headers
        and charts) contain multimodal context that is lost if text and images are embedded in
        isolation. Jointly embedding paired items into a single vector projects both visual and
        textual semantics into a unified representation, boosting downstream retrieval accuracy.
    """
    if not items:
        return []

    with _logfire_span(
        "embed_multimodal",
        item_count=len(items),
        model=model,
        dimension=dimension,
        fallback_model=fallback_model,
    ):
        results: List[List[float]] = []
        for item in items:
            if isinstance(item, str):
                results.append(
                    embed_text(
                        item,
                        model=model,
                        dimension=dimension,
                        fallback_model=fallback_model,
                        normalize=normalize,
                        api_key=api_key,
                    )
                )
            elif isinstance(item, dict) and "image" in item:
                # Paired text + image: co-embed into a joint multimodal vector using multi-part contents
                pair_img = item["image"]
                pair_text = item.get("text", "")
                img_bytes, mime = _to_image_bytes(pair_img)
                client = get_genai_client(api_key=api_key)
                config = types.EmbedContentConfig(
                    task_type=DOCUMENT_TASK_TYPE,
                    output_dimensionality=dimension,
                )
                try:
                    res = client.models.embed_content(
                        model=model,
                        contents=[pair_text, types.Part.from_bytes(
                            data=img_bytes, mime_type=mime)],
                        config=config,
                    )
                    vec = list(res.embeddings[0].values)
                except Exception:
                    # Fallback via unified image fallback
                    vec = _embed_image_fallback(
                        image=pair_img,
                        fallback_model=fallback_model or DEFAULT_FALLBACK_MODEL,
                        dimension=dimension,
                        task_type=DOCUMENT_TASK_TYPE,
                        api_key=api_key,
                    )
                results.append(normalize_vector(vec) if normalize else vec)
            else:
                # Single image
                results.append(
                    embed_image(
                        item,
                        model=model,
                        dimension=dimension,
                        fallback_model=fallback_model,
                        normalize=normalize,
                        api_key=api_key,
                    )
                )
        return results


# ==============================================================================
# ASYNCHRONOUS EMBEDDING FUNCTIONS
# ==============================================================================


@observe()
async def aembed_texts(
    texts: Union[str, List[str]],
    model: str = DEFAULT_PRIMARY_MODEL,
    dimension: Optional[int] = DEFAULT_DIMENSION,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    task_type: str = DOCUMENT_TASK_TYPE,
    fallback_model: Optional[str] = DEFAULT_FALLBACK_MODEL,
    rpm_limit: Optional[int] = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    normalize: bool = True,
    api_key: Optional[str] = None,
    prefer_fallback: bool = False,
) -> List[List[float]]:
    """Asynchronously embed a collection of texts with rate limiting, batching, and Jina AI fallback."""
    if isinstance(texts, str):
        raw_list = [texts]
    else:
        raw_list = list(texts)

    if not raw_list:
        return []

    sanitized_texts = [_sanitize_text(t) for t in raw_list]
    effective_batch_size = min(max(1, batch_size), MAX_GOOGLE_BATCH_SIZE)

    client = get_genai_client(api_key=api_key)
    rate_limiter = get_rate_limiter(rpm=rpm_limit)
    batches = _chunk_list(sanitized_texts, effective_batch_size)
    num_batches = len(batches)
    semaphore = asyncio.Semaphore(max_concurrency)

    with _logfire_span(
        "aembed_texts",
        total_texts=len(sanitized_texts),
        batch_size=effective_batch_size,
        num_batches=num_batches,
        model=model,
        fallback_model=fallback_model,
        dimension=dimension,
        task_type=task_type,
        normalize=normalize,
    ) as root_span:
        async def _process_batch(batch_idx: int, batch_texts: List[str]) -> tuple[int, List[List[float]]]:
            async with semaphore:
                with _logfire_span(
                    "aembed_texts.batch",
                    batch_index=batch_idx + 1,
                    num_batches=num_batches,
                    item_count=len(batch_texts),
                    model=model,
                ):
                    # Route directly to fallback if requested
                    if prefer_fallback or getattr(settings, "USE_FALLOUT_MODEL", False):
                        fallback_embs = await _aembed_fallback_batch(
                            batch=batch_texts,
                            fallback_model=fallback_model or DEFAULT_FALLBACK_MODEL,
                            dimension=dimension,
                            task_type=task_type,
                            timeout=timeout,
                            api_key=api_key,
                        )
                        return batch_idx, fallback_embs

                    try:
                        embs = await _aembed_google_batch_with_retry(
                            client=client,
                            batch=batch_texts,
                            model=model,
                            dimension=dimension,
                            task_type=task_type,
                            max_retries=max_retries,
                            timeout=timeout,
                            rate_limiter=rate_limiter,
                            max_concurrency=max_concurrency,
                        )
                        return batch_idx, embs
                    except Exception as primary_exc:
                        if fallback_model:
                            logger.warning(
                                f"Async primary model '{model}' failed on batch {batch_idx + 1}/{num_batches}: {primary_exc}. "
                                f"Failing over to fallback model '{fallback_model}'..."
                            )
                            if logfire:
                                logfire.warn(
                                    "Async primary model {model} failed on batch {batch_idx}/{num_batches}, failing over to {fallback_model}: {error}",
                                    model=model,
                                    batch_idx=batch_idx + 1,
                                    num_batches=num_batches,
                                    fallback_model=fallback_model,
                                    error=str(primary_exc),
                                )
                            try:
                                fallback_embs = await _aembed_fallback_batch(
                                    batch=batch_texts,
                                    fallback_model=fallback_model,
                                    dimension=dimension,
                                    task_type=task_type,
                                    timeout=timeout,
                                    api_key=api_key,
                                )
                                return batch_idx, fallback_embs
                            except Exception as fallback_exc:
                                logger.error(
                                    f"Async fallback model '{fallback_model}' also failed on batch {batch_idx + 1}: {fallback_exc}"
                                )
                                raise fallback_exc from primary_exc
                        else:
                            raise primary_exc

        tasks = [_process_batch(i, batch) for i, batch in enumerate(batches)]
        completed = await asyncio.gather(*tasks)

        completed_sorted = sorted(completed, key=lambda x: x[0])
        flattened_embeddings: List[List[float]] = []
        for _, batch_embs in completed_sorted:
            flattened_embeddings.extend(batch_embs)

        if normalize:
            flattened_embeddings = [normalize_vector(
                vec) for vec in flattened_embeddings]

        total_chars = sum(len(t) for t in sanitized_texts)
        est_tokens = sum(max(1, max(int(len(t.split()) * 1.3), len(t) // 4)) for t in sanitized_texts)

        if root_span:
            root_span.set_attribute("vector_count", len(flattened_embeddings))
            if flattened_embeddings:
                root_span.set_attribute(
                    "vector_dim", len(flattened_embeddings[0]))
            root_span.set_attribute("tokens.estimated_total", est_tokens)
            root_span.set_attribute("tokens.character_count", total_chars)
            root_span.set_attribute("tokens.avg_tokens_per_item", round(est_tokens / len(sanitized_texts), 2) if sanitized_texts else 0.0)
            root_span.set_attribute("embeddings.vector_count", len(flattened_embeddings))
            root_span.set_attribute("embeddings.dimension", len(flattened_embeddings[0]) if flattened_embeddings else 0)

        return flattened_embeddings


@observe()
async def aembed_text(
    text: str,
    model: str = DEFAULT_PRIMARY_MODEL,
    dimension: Optional[int] = DEFAULT_DIMENSION,
    task_type: str = DOCUMENT_TASK_TYPE,
    fallback_model: Optional[str] = DEFAULT_FALLBACK_MODEL,
    rpm_limit: Optional[int] = None,
    normalize: bool = True,
    api_key: Optional[str] = None,
    prefer_fallback: bool = False,
) -> List[float]:
    """Asynchronously embed a single text string using Google Gemini Embedding 2 with Jina Embeddings v4 fallback."""
    with _logfire_span(
        "aembed_text",
        text_length=len(text) if text else 0,
        model=model,
        dimension=dimension,
        task_type=task_type,
    ):
        results = await aembed_texts(
            texts=[text],
            model=model,
            dimension=dimension,
            batch_size=1,
            max_concurrency=1,
            task_type=task_type,
            fallback_model=fallback_model,
            rpm_limit=rpm_limit,
            normalize=normalize,
            api_key=api_key,
            prefer_fallback=prefer_fallback,
        )
        return results[0] if results else []


@observe()
async def aembed_query(
    query: str,
    model: str = DEFAULT_PRIMARY_MODEL,
    dimension: Optional[int] = DEFAULT_DIMENSION,
    fallback_model: Optional[str] = DEFAULT_FALLBACK_MODEL,
    rpm_limit: Optional[int] = None,
    normalize: bool = True,
    api_key: Optional[str] = None,
    prefer_fallback: bool = False,
) -> List[float]:
    """Asynchronously embed a search query using Google Gemini Embedding 2 with task_type='RETRIEVAL_QUERY'."""
    with _logfire_span(
        "aembed_query",
        query_length=len(query) if query else 0,
        model=model,
        dimension=dimension,
        fallback_model=fallback_model,
        prefer_fallback=prefer_fallback,
    ):
        return await aembed_text(
            text=query,
            model=model,
            dimension=dimension,
            task_type=QUERY_TASK_TYPE,
            fallback_model=fallback_model,
            rpm_limit=rpm_limit,
            normalize=normalize,
            api_key=api_key,
            prefer_fallback=prefer_fallback,
        )


@observe()
async def aembed_image(
    image: ImageInput,
    model: str = DEFAULT_PRIMARY_MODEL,
    dimension: Optional[int] = DEFAULT_DIMENSION,
    task_type: str = DOCUMENT_TASK_TYPE,
    fallback_model: Optional[str] = DEFAULT_FALLBACK_MODEL,
    rpm_limit: Optional[int] = None,
    normalize: bool = True,
    api_key: Optional[str] = None,
    prefer_fallback: bool = False,
) -> List[float]:
    """Asynchronously embed an image using Google Gemini Embedding 2 with Jina Embeddings v4 fallback."""
    with _logfire_span(
        "aembed_image",
        model=model,
        dimension=dimension,
        task_type=task_type,
        fallback_model=fallback_model,
        prefer_fallback=prefer_fallback,
    ):
        if prefer_fallback or getattr(settings, "USE_FALLOUT_MODEL", False):
            vec = await _aembed_image_fallback(
                image=image,
                fallback_model=fallback_model or DEFAULT_FALLBACK_MODEL,
                dimension=dimension,
                task_type=task_type,
                api_key=api_key,
            )
            return normalize_vector(vec) if normalize else vec

        client = get_genai_client(api_key=api_key)
        rate_limiter = get_rate_limiter(rpm=rpm_limit)
        await rate_limiter.acquire_async(1)

        img_bytes, mime = _to_image_bytes(image)

        config_kwargs: dict[str, Any] = {
            "http_options": types.HttpOptions(timeout=int(DEFAULT_TIMEOUT_SECONDS * 1000))
        }
        if dimension is not None:
            config_kwargs["output_dimensionality"] = dimension
        if task_type:
            config_kwargs["task_type"] = task_type
        config = types.EmbedContentConfig(**config_kwargs)

        try:
            res = await client.aio.models.embed_content(
                model=model,
                contents=[types.Part.from_bytes(
                    data=img_bytes, mime_type=mime)],
                config=config,
            )
            vec = list(res.embeddings[0].values)
        except Exception as primary_exc:
            logger.warning(
                f"Async primary image embedding with '{model}' failed: {primary_exc}. "
                f"Failing over to fallback model '{fallback_model}'..."
            )
            if logfire:
                logfire.warn(
                    "Async primary image embedding with {model} failed, failing over to {fallback_model}: {error}",
                    model=model,
                    fallback_model=fallback_model,
                    error=str(primary_exc),
                )
            try:
                vec = await _aembed_image_fallback(
                    image=image,
                    fallback_model=fallback_model or DEFAULT_FALLBACK_MODEL,
                    dimension=dimension,
                    task_type=task_type,
                    api_key=api_key,
                )
            except Exception as fallback_exc:
                logger.error(
                    f"Async multimodal image fallback '{fallback_model}' also failed: {fallback_exc}")
                raise fallback_exc from primary_exc

        return normalize_vector(vec) if normalize else vec


# ==============================================================================
# LANGCHAIN EMBEDDINGS PROTOCOL ADAPTER
# ==============================================================================


@observe()
class GeminiMultimodalEmbeddings(Embeddings):
    """LangChain-compatible Embeddings adapter using Google Gemini Embedding 2 with automatic Jina AI fallback.

    What:
        Implements the standard LangChain Embeddings interface (embed_documents, embed_query,
        aembed_documents, aembed_query) atop the production-grade Gemini and Jina embeddings service.
        Respects RPM token-bucket limits, MRL dimensions, asymmetric query/passage heads, and
        transparent automatic failover to the fallout model.

    Why:
        Allows direct drop-in integration with any LangChain vector store (e.g. Qdrant, Chroma, PGVector),
        LCEL chains, retrievers, and agentic workflows while maintaining resilient fallback handling.
    """

    def __init__(
        self,
        model: str = DEFAULT_PRIMARY_MODEL,
        fallback_model: Optional[str] = DEFAULT_FALLBACK_MODEL,
        dimension: Optional[int] = DEFAULT_DIMENSION,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        rpm_limit: Optional[int] = None,
        prefer_fallback: bool = False,
        normalize: bool = True,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.fallback_model = fallback_model
        self.dimension = dimension
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency
        self.rpm_limit = rpm_limit
        self.prefer_fallback = prefer_fallback or getattr(
            settings, "USE_FALLOUT_MODEL", False)
        self.normalize = normalize
        self.api_key = api_key

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed document passages for ingestion using RETRIEVAL_DOCUMENT task."""
        return embed_texts(
            texts=texts,
            model=self.model,
            dimension=self.dimension,
            batch_size=self.batch_size,
            max_concurrency=self.max_concurrency,
            task_type=DOCUMENT_TASK_TYPE,
            fallback_model=self.fallback_model,
            rpm_limit=self.rpm_limit,
            normalize=self.normalize,
            api_key=self.api_key,
            prefer_fallback=self.prefer_fallback,
        )

    def embed_query(self, text: str) -> List[float]:
        """Embed search query for retrieval using RETRIEVAL_QUERY task."""
        return embed_query(
            query=text,
            model=self.model,
            dimension=self.dimension,
            fallback_model=self.fallback_model,
            rpm_limit=self.rpm_limit,
            normalize=self.normalize,
            api_key=self.api_key,
            prefer_fallback=self.prefer_fallback,
        )

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously embed document passages for ingestion."""
        return await aembed_texts(
            texts=texts,
            model=self.model,
            dimension=self.dimension,
            batch_size=self.batch_size,
            max_concurrency=self.max_concurrency,
            task_type=DOCUMENT_TASK_TYPE,
            fallback_model=self.fallback_model,
            rpm_limit=self.rpm_limit,
            normalize=self.normalize,
            api_key=self.api_key,
            prefer_fallback=self.prefer_fallback,
        )

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronously embed search query for retrieval."""
        return await aembed_query(
            query=text,
            model=self.model,
            dimension=self.dimension,
            fallback_model=self.fallback_model,
            rpm_limit=self.rpm_limit,
            normalize=self.normalize,
            api_key=self.api_key,
            prefer_fallback=self.prefer_fallback,
        )


@observe()
def retrieve_query_vector(
    query: str,
    prefer_fallback: bool = False,
    model: str = DEFAULT_PRIMARY_MODEL,
    fallback_model: Optional[str] = DEFAULT_FALLBACK_MODEL,
    dimension: Optional[int] = DEFAULT_DIMENSION,
) -> List[float]:
    """Convenience helper to embed a retrieval query vector with explicit fallback/fallout model support."""
    return embed_query(
        query=query,
        model=model,
        dimension=dimension,
        fallback_model=fallback_model,
        prefer_fallback=prefer_fallback,
    )

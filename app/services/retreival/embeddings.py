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

from app.config import settings

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
_GENAI_CLIENT: Optional[Any] = None
_JINA_CLIENT: Optional[Any] = None
_ASYNC_JINA_CLIENT: Optional[Any] = None
_RATE_LIMITERS: Dict[int, "TokenBucketRateLimiter"] = {}
_RATE_LIMITER_LOCK = threading.Lock()

# Supported image input types
ImageInput = Union[Image.Image, bytes, bytearray, str, Path, Any]


# ==============================================================================
# TOKEN BUCKET RATE LIMITER (PROACTIVE RATE LIMITING)
# ==============================================================================


class TokenBucketRateLimiter:
    """Thread-safe and async-safe Token Bucket Rate Limiter.

    Proactively controls the request dispatch rate to stay within Google Gemini's
    Requests-Per-Minute (RPM) limits and avoid 429 Too Many Requests errors.
    """

    def __init__(self, rpm: int = DEFAULT_RPM_LIMIT, burst_size: Optional[int] = None):
        self.rpm = max(1, rpm)
        self.rate = self.rpm / 60.0  # tokens added per second
        self.capacity = burst_size or max(1, min(self.rpm, int(self.rpm / 6)))  # burst allowance
        self.tokens = float(self.capacity)
        self.last_update = time.monotonic()
        self._lock = threading.Lock()

    def _replenish(self, now: float) -> None:
        elapsed = now - self.last_update
        self.tokens = min(float(self.capacity), self.tokens + elapsed * self.rate)
        self.last_update = now

    def acquire(self, tokens: int = 1) -> None:
        """Synchronously acquire tokens, sleeping if the bucket is empty."""
        while True:
            with self._lock:
                now = time.monotonic()
                self._replenish(now)
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return
                needed = tokens - self.tokens
                sleep_time = needed / self.rate
            time.sleep(min(sleep_time, 1.0))

    async def acquire_async(self, tokens: int = 1) -> None:
        """Asynchronously acquire tokens without blocking the event loop."""
        while True:
            now = time.monotonic()
            with self._lock:
                self._replenish(now)
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    return
                needed = tokens - self.tokens
                sleep_time = needed / self.rate
            await asyncio.sleep(min(sleep_time, 1.0))

    def throttle(self, penalty_seconds: float) -> None:
        """Temporarily throttle the bucket when an external 429 response is encountered."""
        with self._lock:
            self.tokens = -penalty_seconds * self.rate
            self.last_update = time.monotonic()


def get_rate_limiter(rpm: Optional[int] = None) -> TokenBucketRateLimiter:
    """Retrieve or create a cached TokenBucketRateLimiter for the given RPM."""
    target_rpm = rpm or getattr(settings, "EMBEDDING_RPM_LIMIT", DEFAULT_RPM_LIMIT) or DEFAULT_RPM_LIMIT
    with _RATE_LIMITER_LOCK:
        if target_rpm not in _RATE_LIMITERS:
            _RATE_LIMITERS[target_rpm] = TokenBucketRateLimiter(rpm=target_rpm)
        return _RATE_LIMITERS[target_rpm]


def _extract_retry_delay(exc: Exception, default_delay: float) -> float:
    """Extract retry-after seconds from 429 error messages or headers if present."""
    msg = str(exc).lower()
    match = re.search(r"retry[- ]after[:\s]+([0-9.]+)", msg)
    if not match:
        match = re.search(r"seconds[:\s]+([0-9.]+)", msg)
    if not match:
        match = re.search(r"wait\s+([0-9.]+)\s*s", msg)

    if match:
        try:
            return float(match.group(1)) + random.uniform(0.2, 0.8)
        except (ValueError, TypeError):
            pass
    return default_delay


# ==============================================================================
# CLIENT MANAGEMENT
# ==============================================================================


def get_genai_client(api_key: Optional[str] = None) -> genai.Client:
    """Get or initialize a cached Google GenAI Client."""
    global _GENAI_CLIENT
    key = api_key or settings.GEMINI_API_KEY
    if not key:
        raise ValueError(
            "GEMINI_API_KEY is not set. Please provide it in your environment or config."
        )

    if _GENAI_CLIENT is None or api_key is not None:
        client = genai.Client(api_key=key)
        if api_key is None:
            _GENAI_CLIENT = client
        return client
    return _GENAI_CLIENT


def _get_jina_api_key(api_key: Optional[str] = None) -> Optional[str]:
    """Retrieve Jina AI API key from explicit argument, settings, or environment."""
    return (
        api_key
        or getattr(settings, "JINA_API_KEY", None)
        or os.getenv("JINA_API_KEY")
        or os.getenv("JINA_AI_API_TOKEN")
    )


def get_jina_client(timeout: float = DEFAULT_TIMEOUT_SECONDS) -> httpx.Client:
    """Get or initialize a cached synchronous httpx.Client for Jina AI API calls."""
    global _JINA_CLIENT
    if httpx is None:
        raise ImportError(
            "httpx is required for Jina AI embeddings. Install it via 'pip install httpx'."
        )
    if _JINA_CLIENT is None or _JINA_CLIENT.is_closed:
        _JINA_CLIENT = httpx.Client(timeout=timeout)
    return _JINA_CLIENT


def get_async_jina_client(timeout: float = DEFAULT_TIMEOUT_SECONDS) -> httpx.AsyncClient:
    """Get or initialize a cached asynchronous httpx.AsyncClient for Jina AI API calls."""
    global _ASYNC_JINA_CLIENT
    if httpx is None:
        raise ImportError(
            "httpx is required for Jina AI embeddings. Install it via 'pip install httpx'."
        )
    if _ASYNC_JINA_CLIENT is None or _ASYNC_JINA_CLIENT.is_closed:
        _ASYNC_JINA_CLIENT = httpx.AsyncClient(timeout=timeout)
    return _ASYNC_JINA_CLIENT


# ==============================================================================
# VECTOR NORMALIZATION & IMAGE UTILITIES
# ==============================================================================


def normalize_vector(vector: Sequence[float]) -> List[float]:
    """L2-normalizes an embedding vector for exact cosine similarity."""
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0.0 or math.isnan(norm):
        return list(vector)
    return [x / norm for x in vector]


def _chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Partition a list into sublists of size at most chunk_size."""
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def _sanitize_text(text: str) -> str:
    """Ensure texts are non-empty strings so embedding models do not reject them."""
    if not isinstance(text, str):
        text = str(text)
    stripped = text.strip()
    return stripped if stripped else " "


def _to_image_bytes(item: ImageInput) -> tuple[bytes, str]:
    """Convert an image input (PIL Image, bytes, Path, Base64 data URI, ExtractedImage) to (bytes, mime_type)."""
    # 1. ExtractedImage instance from load.py
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
            header, b64_data = item.split(",", 1)
            mime = header.split(";")[0].replace("data:", "")
            return base64.b64decode(b64_data), mime or "image/png"

    raise ValueError(f"Unsupported image input type for embedding: {type(item)}")


def _to_image_data_uri(item: ImageInput) -> str:
    """Convert an image input to a Base64 data URI string (data:image/png;base64,...)."""
    if isinstance(item, str) and item.startswith("data:image/"):
        return item
    img_bytes, mime = _to_image_bytes(item)
    b64_str = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64_str}"


def _map_task_to_jina(task_type: Optional[str]) -> str:
    """Map Google or generic task type to Jina AI task adapter.

    Supported Jina tasks:
    - 'retrieval.passage' (indexing document chunks)
    - 'retrieval.query' (search queries)
    - 'text-matching' (semantic similarity / classification)
    """
    if not task_type:
        return "retrieval.passage"
    t = task_type.lower()
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
    """Execute batch embedding against Google GenAI with proactive rate limiting & retry.

    For Gemini Embedding 2 ('gemini-embedding-2'), passing multiple items generates
    a single aggregated embedding. To generate distinct 1-to-1 embeddings for each chunk,
    batch items are embedded individually with thread-pool concurrency and rate limiting.
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

    if is_gemini_2 and len(batch) > 1:
        def _embed_single(text: str) -> List[float]:
            last_err = None
            for attempt in range(1, max_retries + 1):
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
                    if "404" in err_str or "not found" in err_str or "400" in err_str:
                        raise e

                    if attempt < max_retries:
                        backoff = (2 ** (attempt - 1)) + random.uniform(0.2, 0.6)
                        if "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str:
                            backoff = _extract_retry_delay(e, default_delay=backoff * 2.0)
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

    contents_input = batch[0] if (is_gemini_2 and len(batch) == 1) else batch

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
                logger.warning(f"Google embedding model '{model}' non-retryable error: {e}")
                raise e

            if attempt < max_retries:
                backoff = (2 ** (attempt - 1)) + random.uniform(0.2, 0.6)
                if "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str:
                    backoff = _extract_retry_delay(e, default_delay=backoff * 2.0)
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
    """Execute batch embedding asynchronously with proactive rate limiting & retry."""
    config_kwargs: dict[str, Any] = {
        "http_options": types.HttpOptions(timeout=int(timeout * 1000))
    }
    if dimension is not None:
        config_kwargs["output_dimensionality"] = dimension
    if task_type:
        config_kwargs["task_type"] = task_type

    config = types.EmbedContentConfig(**config_kwargs)
    is_gemini_2 = "gemini-embedding-2" in model.lower()

    if is_gemini_2 and len(batch) > 1:
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
                            backoff = (2 ** (attempt - 1)) + random.uniform(0.2, 0.6)
                            if "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str:
                                backoff = _extract_retry_delay(e, default_delay=backoff * 2.0)
                                rate_limiter.throttle(backoff)
                                logger.warning(
                                    f"Async rate limit (429) hit for '{model}'. Throttling and retrying in {backoff:.2f}s..."
                                )
                            await asyncio.sleep(backoff)
                        else:
                            raise e
                raise last_err or RuntimeError("Single item async embedding failed")

        tasks = [_aembed_single(t) for t in batch]
        return await asyncio.gather(*tasks)

    contents_input = batch[0] if (is_gemini_2 and len(batch) == 1) else batch

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
                logger.warning(f"Google embedding model '{model}' non-retryable error: {e}")
                raise e

            if attempt < max_retries:
                backoff = (2 ** (attempt - 1)) + random.uniform(0.2, 0.6)
                if "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str:
                    backoff = _extract_retry_delay(e, default_delay=backoff * 2.0)
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

    Features:
    - Native multimodal support: accepts text strings, structured text dicts,
      and Base64 data URIs for images.
    - Matryoshka Representation Learning (MRL): truncates embeddings to exact
      dimension (default: 768) to maintain vector database schema alignment.
    - Automatic task adapter selection: 'retrieval.passage', 'retrieval.query',
      or 'text-matching'.
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
    if dimension is not None:
        payload["dimensions"] = dimension

    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.post(JINA_API_URL, headers=headers, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("data", [])
                sorted_items = sorted(items, key=lambda x: x.get("index", 0))
                return [item["embedding"] for item in sorted_items]

            status = resp.status_code
            if status == 429:
                retry_after = float(resp.headers.get("Retry-After", 2.0))
                backoff = max(retry_after, (2 ** (attempt - 1)) + random.uniform(0.2, 0.8))
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

    raise last_exc or RuntimeError(f"Failed to embed batch with Jina AI '{model}'")


async def _aembed_jina_batch(
    inputs: List[Union[str, Dict[str, Any]]],
    model: str = DEFAULT_FALLBACK_MODEL,
    dimension: Optional[int] = DEFAULT_DIMENSION,
    task_type: Optional[str] = DOCUMENT_TASK_TYPE,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    api_key: Optional[str] = None,
) -> List[List[float]]:
    """Asynchronously embed a batch of inputs using Jina AI API (jina-embeddings-v4)."""
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
    if dimension is not None:
        payload["dimensions"] = dimension

    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = await client.post(JINA_API_URL, headers=headers, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("data", [])
                sorted_items = sorted(items, key=lambda x: x.get("index", 0))
                return [item["embedding"] for item in sorted_items]

            status = resp.status_code
            if status == 429:
                retry_after = float(resp.headers.get("Retry-After", 2.0))
                backoff = max(retry_after, (2 ** (attempt - 1)) + random.uniform(0.2, 0.8))
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

    raise last_exc or RuntimeError(f"Async embedding failed with Jina AI '{model}'")


# ==============================================================================
# FALLBACK EMBEDDING ROUTERS (JINA EMBEDDINGS V4)
# ==============================================================================


def _embed_fallback_batch(
    batch: List[str],
    fallback_model: str = DEFAULT_FALLBACK_MODEL,
    dimension: Optional[int] = DEFAULT_DIMENSION,
    task_type: Optional[str] = DOCUMENT_TASK_TYPE,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    api_key: Optional[str] = None,
) -> List[List[float]]:
    """Route batch text embedding to the fallback model (jina-embeddings-v4)."""
    return _embed_jina_batch(
        inputs=batch,
        model=fallback_model,
        dimension=dimension,
        task_type=task_type,
        timeout=timeout,
        max_retries=max_retries,
        api_key=api_key,
    )


async def _aembed_fallback_batch(
    batch: List[str],
    fallback_model: str = DEFAULT_FALLBACK_MODEL,
    dimension: Optional[int] = DEFAULT_DIMENSION,
    task_type: Optional[str] = DOCUMENT_TASK_TYPE,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = DEFAULT_MAX_RETRIES,
    api_key: Optional[str] = None,
) -> List[List[float]]:
    """Asynchronously route batch text embedding to the fallback model (jina-embeddings-v4)."""
    return await _aembed_jina_batch(
        inputs=batch,
        model=fallback_model,
        dimension=dimension,
        task_type=task_type,
        timeout=timeout,
        max_retries=max_retries,
        api_key=api_key,
    )


def _embed_image_fallback(
    image: ImageInput,
    fallback_model: str = DEFAULT_FALLBACK_MODEL,
    dimension: Optional[int] = DEFAULT_DIMENSION,
    task_type: Optional[str] = DOCUMENT_TASK_TYPE,
    api_key: Optional[str] = None,
) -> List[float]:
    """Route image embedding to the fallback model (jina-embeddings-v4) using native image input."""
    data_uri = _to_image_data_uri(image)
    embs = _embed_jina_batch(
        inputs=[{"image": data_uri}],
        model=fallback_model,
        dimension=dimension,
        task_type=task_type,
        api_key=api_key,
    )
    return embs[0]


async def _aembed_image_fallback(
    image: ImageInput,
    fallback_model: str = DEFAULT_FALLBACK_MODEL,
    dimension: Optional[int] = DEFAULT_DIMENSION,
    task_type: Optional[str] = DOCUMENT_TASK_TYPE,
    api_key: Optional[str] = None,
) -> List[float]:
    """Asynchronously route image embedding to the fallback model (jina-embeddings-v4) using native image input."""
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
) -> List[List[float]]:
    """Embed texts using Google Gemini Embedding 2 with rate limiting, batching, and Jina Embeddings v4 fallback.

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

    Returns:
        List[List[float]]: List of embedding vectors in 1-to-1 correspondence with texts.
    """
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

    logger.debug(
        f"Embedding {len(sanitized_texts)} texts into {num_batches} batches "
        f"(batch_size={effective_batch_size}, dimension={dimension}, model='{model}', fallback='{fallback_model}')"
    )

    def _process_batch(batch_idx: int, batch_texts: List[str]) -> tuple[int, List[List[float]]]:
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

    ordered_results: List[Optional[List[List[float]]]] = [None] * num_batches

    if num_batches == 1:
        _, ordered_results[0] = _process_batch(0, batches[0])
    else:
        concurrency = min(max_concurrency, num_batches)
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_idx = {
                executor.submit(_process_batch, i, batch): i for i, batch in enumerate(batches)
            }
            for future in as_completed(future_to_idx):
                idx, embs = future.result()
                ordered_results[idx] = embs

    flattened_embeddings: List[List[float]] = []
    for batch_embs in ordered_results:
        if batch_embs is not None:
            flattened_embeddings.extend(batch_embs)

    if normalize:
        flattened_embeddings = [normalize_vector(vec) for vec in flattened_embeddings]

    return flattened_embeddings


def embed_text(
    text: str,
    model: str = DEFAULT_PRIMARY_MODEL,
    dimension: Optional[int] = DEFAULT_DIMENSION,
    task_type: str = DOCUMENT_TASK_TYPE,
    fallback_model: Optional[str] = DEFAULT_FALLBACK_MODEL,
    rpm_limit: Optional[int] = None,
    normalize: bool = True,
    api_key: Optional[str] = None,
) -> List[float]:
    """Embed a single text string using Google Gemini Embedding 2 with Jina Embeddings v4 fallback."""
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
    )
    return results[0] if results else []


def embed_query(
    query: str,
    model: str = DEFAULT_PRIMARY_MODEL,
    dimension: Optional[int] = DEFAULT_DIMENSION,
    fallback_model: Optional[str] = DEFAULT_FALLBACK_MODEL,
    rpm_limit: Optional[int] = None,
    normalize: bool = True,
    api_key: Optional[str] = None,
) -> List[float]:
    """Embed a search query using Google Gemini Embedding 2 with task_type='RETRIEVAL_QUERY'."""
    return embed_text(
        text=query,
        model=model,
        dimension=dimension,
        task_type=QUERY_TASK_TYPE,
        fallback_model=fallback_model,
        rpm_limit=rpm_limit,
        normalize=normalize,
        api_key=api_key,
    )


# ==============================================================================
# MULTIMODAL & IMAGE EMBEDDING FUNCTIONS
# ==============================================================================


def embed_image(
    image: ImageInput,
    model: str = DEFAULT_PRIMARY_MODEL,
    dimension: Optional[int] = DEFAULT_DIMENSION,
    task_type: str = DOCUMENT_TASK_TYPE,
    fallback_model: Optional[str] = DEFAULT_FALLBACK_MODEL,
    rpm_limit: Optional[int] = None,
    normalize: bool = True,
    api_key: Optional[str] = None,
) -> List[float]:
    """Embed an image using Google Gemini Embedding 2 with Jina Embeddings v4 multimodal fallback.

    Accepts:
    - PIL Image.Image instances
    - Raw image bytes
    - File path string or Path object
    - Base64 data URI string
    - ExtractedImage object (from app.ingestion.loader.load)

    Args:
        image: Target image representation.
        model: Primary embedding model ('gemini-embedding-2').
        dimension: Desired embedding dimension (default: 768).
        task_type: Retrieval task type (default: 'RETRIEVAL_DOCUMENT').
        fallback_model: Fallback model ('jina-embeddings-v4').
        rpm_limit: Requests-per-minute throttle.
        normalize: Whether to L2-normalize vectors (default: True).
        api_key: Optional Gemini API key override.

    Returns:
        List[float]: 768-dimensional embedding vector representing the visual content.
    """
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
        res = client.models.embed_content(
            model=model,
            contents=[types.Part.from_bytes(data=img_bytes, mime_type=mime)],
            config=config,
        )
        vec = list(res.embeddings[0].values)
    except Exception as primary_exc:
        logger.warning(
            f"Primary image embedding with '{model}' failed: {primary_exc}. "
            f"Failing over to fallback model '{fallback_model}'..."
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
            logger.error(f"Multimodal image fallback '{fallback_model}' also failed: {fallback_exc}")
            raise fallback_exc from primary_exc

    return normalize_vector(vec) if normalize else vec


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
) -> List[List[float]]:
    """Concurrently embed a collection of images using Gemini Embedding 2 with Jina Embeddings v4 fallback."""
    if not images:
        return []

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
            )
            for img in images
        ]
        return [f.result() for f in futures]


def embed_multimodal(
    items: Sequence[Union[str, ImageInput, Dict[str, Any]]],
    model: str = DEFAULT_PRIMARY_MODEL,
    dimension: Optional[int] = DEFAULT_DIMENSION,
    fallback_model: Optional[str] = DEFAULT_FALLBACK_MODEL,
    normalize: bool = True,
    api_key: Optional[str] = None,
) -> List[List[float]]:
    """Embed mixed collections of text strings, images, or paired multimodal dicts."""
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
            # Paired text + image
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
                    contents=[pair_text, types.Part.from_bytes(data=img_bytes, mime_type=mime)],
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
) -> List[List[float]]:
    """Asynchronously embed texts with Google Gemini Embedding 2, rate limiting, and Jina Embeddings v4 fallback."""
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

    async def _process_batch(batch_idx: int, batch_texts: List[str]) -> tuple[int, List[List[float]]]:
        async with semaphore:
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
        flattened_embeddings = [normalize_vector(vec) for vec in flattened_embeddings]

    return flattened_embeddings


async def aembed_text(
    text: str,
    model: str = DEFAULT_PRIMARY_MODEL,
    dimension: Optional[int] = DEFAULT_DIMENSION,
    task_type: str = DOCUMENT_TASK_TYPE,
    fallback_model: Optional[str] = DEFAULT_FALLBACK_MODEL,
    rpm_limit: Optional[int] = None,
    normalize: bool = True,
    api_key: Optional[str] = None,
) -> List[float]:
    """Asynchronously embed a single text string using Google Gemini Embedding 2 with Jina Embeddings v4 fallback."""
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
    )
    return results[0] if results else []


async def aembed_query(
    query: str,
    model: str = DEFAULT_PRIMARY_MODEL,
    dimension: Optional[int] = DEFAULT_DIMENSION,
    fallback_model: Optional[str] = DEFAULT_FALLBACK_MODEL,
    rpm_limit: Optional[int] = None,
    normalize: bool = True,
    api_key: Optional[str] = None,
) -> List[float]:
    """Asynchronously embed a search query using Google Gemini Embedding 2 with task_type='RETRIEVAL_QUERY'."""
    return await aembed_text(
        text=query,
        model=model,
        dimension=dimension,
        task_type=QUERY_TASK_TYPE,
        fallback_model=fallback_model,
        rpm_limit=rpm_limit,
        normalize=normalize,
        api_key=api_key,
    )


async def aembed_image(
    image: ImageInput,
    model: str = DEFAULT_PRIMARY_MODEL,
    dimension: Optional[int] = DEFAULT_DIMENSION,
    task_type: str = DOCUMENT_TASK_TYPE,
    fallback_model: Optional[str] = DEFAULT_FALLBACK_MODEL,
    rpm_limit: Optional[int] = None,
    normalize: bool = True,
    api_key: Optional[str] = None,
) -> List[float]:
    """Asynchronously embed an image using Google Gemini Embedding 2 with Jina Embeddings v4 fallback."""
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
            contents=[types.Part.from_bytes(data=img_bytes, mime_type=mime)],
            config=config,
        )
        vec = list(res.embeddings[0].values)
    except Exception as primary_exc:
        logger.warning(
            f"Async primary image embedding with '{model}' failed: {primary_exc}. "
            f"Failing over to fallback model '{fallback_model}'..."
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
            logger.error(f"Async multimodal image fallback '{fallback_model}' also failed: {fallback_exc}")
            raise fallback_exc from primary_exc

    return normalize_vector(vec) if normalize else vec

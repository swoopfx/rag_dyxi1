"""Embeddings service using Google Gemini Embedding 2 for vector database ingestion.

Features:
- Primary model: Google Gemini Embedding 2 ('gemini-embedding-2').
- Fallback models: 'gemini-embedding-001' or OpenAI 'text-embedding-3-small'.
- Proactive & Reactive Rate Limiting:
  * Thread-safe and async-safe Token Bucket rate limiter (configurable RPM).
  * 429 / ResourceExhausted detection with 'retry-after' delay extraction.
  * Shared throttling across concurrent worker threads to prevent cascade rate limits.
  * Automatic failover to fallback models upon quota / rate exhaustion.
- Automatic batching with configurable batch size.
- 1-to-1 vector generation: handles Gemini Embedding 2's aggregation behavior by
  concurrently embedding individual chunks in parallel batches.
- Dimension control via Matryoshka Representation Learning (MRL) (e.g. 768, 512, 256).
- Low latency via concurrent parallel batch requests and client connection pooling.
- Automatic retry with exponential backoff & jitter for transient errors.
- Synchronous and asynchronous embedding functions for ingestion and search.
"""

from __future__ import annotations

import asyncio
import math
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence, Union

from loguru import logger

from app.config import settings

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

# OpenAI fallback import
try:
    from openai import AsyncOpenAI, OpenAI
except ImportError:
    OpenAI = None  # type: ignore
    AsyncOpenAI = None  # type: ignore


# ==============================================================================
# CONFIGURATION DEFAULTS
# ==============================================================================

DEFAULT_PRIMARY_MODEL: str = "gemini-embedding-2"
DEFAULT_FALLBACK_MODEL: str = "gemini-embedding-001"
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
_OPENAI_CLIENT: Optional[Any] = None
_ASYNC_OPENAI_CLIENT: Optional[Any] = None
_RATE_LIMITERS: Dict[int, "TokenBucketRateLimiter"] = {}
_RATE_LIMITER_LOCK = threading.Lock()


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
            # Drain tokens to enforce an immediate pause across all threads
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
    # Match patterns like 'retry after 12.5s', 'retry after 5 seconds', or 'seconds: 10'
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


def get_openai_client(api_key: Optional[str] = None) -> Optional[Any]:
    """Get or initialize a cached OpenAI client for fallback embeddings."""
    global _OPENAI_CLIENT
    if OpenAI is None:
        return None
    key = api_key or settings.OPENAI_API_KEY
    if not key:
        return None

    if _OPENAI_CLIENT is None or api_key is not None:
        client = OpenAI(api_key=key)
        if api_key is None:
            _OPENAI_CLIENT = client
        return client
    return _OPENAI_CLIENT


def get_async_openai_client(api_key: Optional[str] = None) -> Optional[Any]:
    """Get or initialize a cached AsyncOpenAI client for fallback embeddings."""
    global _ASYNC_OPENAI_CLIENT
    if AsyncOpenAI is None:
        return None
    key = api_key or settings.OPENAI_API_KEY
    if not key:
        return None

    if _ASYNC_OPENAI_CLIENT is None or api_key is not None:
        client = AsyncOpenAI(api_key=key)
        if api_key is None:
            _ASYNC_OPENAI_CLIENT = client
        return client
    return _ASYNC_OPENAI_CLIENT


# ==============================================================================
# VECTOR NORMALIZATION & UTILITIES
# ==============================================================================


def normalize_vector(vector: Sequence[float]) -> List[float]:
    """L2-normalizes an embedding vector for exact cosine similarity."""
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0.0 or math.isnan(norm):
        return list(vector)
    return [x / norm for x in vector]


def _chunk_list(lst: List[str], chunk_size: int) -> List[List[str]]:
    """Partition a list into sublists of size at most chunk_size."""
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def _sanitize_text(text: str) -> str:
    """Ensure texts are non-empty strings so embedding models do not reject them."""
    if not isinstance(text, str):
        text = str(text)
    stripped = text.strip()
    return stripped if stripped else " "


# ==============================================================================
# SINGLE-BATCH EMBEDDING WITH PROACTIVE & REACTIVE RATE LIMITING
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

    # Gemini Embedding 2: embed items individually in parallel to ensure 1-to-1 vectors
    if is_gemini_2 and len(batch) > 1:
        def _embed_single(text: str) -> List[float]:
            last_err = None
            for attempt in range(1, max_retries + 1):
                # Proactive rate limiting: acquire permit before firing request
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

    # Single item or batch-supporting model (e.g. gemini-embedding-001)
    contents_input = batch[0] if (is_gemini_2 and len(batch) == 1) else batch

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        # Proactive rate limit acquisition
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
                    # Proactive async rate limiting
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
# FALLBACK EMBEDDING DISPATCHER
# ==============================================================================


def _embed_fallback_batch(
    batch: List[str],
    fallback_model: str,
    dimension: Optional[int],
    task_type: Optional[str],
    genai_client: genai.Client,
    timeout: float,
    rate_limiter: TokenBucketRateLimiter,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> List[List[float]]:
    """Execute embedding on the fallback model (secondary Google model or OpenAI)."""
    logger.info(f"Invoking fallback embedding model: '{fallback_model}' for {len(batch)} items")

    # Path A: Fallback to OpenAI (e.g. text-embedding-3-small)
    if "text-embedding-3" in fallback_model or "openai" in fallback_model.lower():
        openai_client = get_openai_client()
        if openai_client is None:
            raise RuntimeError(
                f"OpenAI fallback model '{fallback_model}' was requested, but OPENAI_API_KEY is not configured."
            )
        kwargs: dict[str, Any] = {"model": fallback_model, "input": batch}
        if dimension is not None:
            kwargs["dimensions"] = dimension
        resp = openai_client.embeddings.create(**kwargs)
        return [item.embedding for item in resp.data]

    # Path B: Fallback to Google model (e.g. gemini-embedding-001)
    return _embed_google_batch_with_retry(
        client=genai_client,
        batch=batch,
        model=fallback_model,
        dimension=dimension,
        task_type=task_type,
        max_retries=2,
        timeout=timeout,
        rate_limiter=rate_limiter,
        max_concurrency=max_concurrency,
    )


async def _aembed_fallback_batch(
    batch: List[str],
    fallback_model: str,
    dimension: Optional[int],
    task_type: Optional[str],
    genai_client: genai.Client,
    timeout: float,
    rate_limiter: TokenBucketRateLimiter,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> List[List[float]]:
    """Execute async embedding on the fallback model."""
    logger.info(f"Invoking async fallback embedding model: '{fallback_model}' for {len(batch)} items")

    if "text-embedding-3" in fallback_model or "openai" in fallback_model.lower():
        async_openai = get_async_openai_client()
        if async_openai is None:
            raise RuntimeError(
                f"OpenAI fallback model '{fallback_model}' was requested, but OPENAI_API_KEY is not configured."
            )
        kwargs: dict[str, Any] = {"model": fallback_model, "input": batch}
        if dimension is not None:
            kwargs["dimensions"] = dimension
        resp = await async_openai.embeddings.create(**kwargs)
        return [item.embedding for item in resp.data]

    return await _aembed_google_batch_with_retry(
        client=genai_client,
        batch=batch,
        model=fallback_model,
        dimension=dimension,
        task_type=task_type,
        max_retries=2,
        timeout=timeout,
        rate_limiter=rate_limiter,
        max_concurrency=max_concurrency,
    )


# ==============================================================================
# MAIN EMBEDDING FUNCTIONS
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
    """Embed texts using Google Gemini Embedding 2 with rate limiting, batching, and fallback.

    Designed for embedding data to be stored in vector databases (e.g., Qdrant).
    Features:
    - Primary model: Google Gemini Embedding 2 ('gemini-embedding-2').
    - Rate Limiting: Proactive token bucket limiter prevents 429 quota exhaustion;
      reactive backoff dynamically throttles parallel workers when rate-limited.
    - Distinct 1-to-1 vector generation: handles Gemini Embedding 2's aggregation
      behavior by embedding individual items concurrently across batches.
    - Concurrency reduces latency by running batch requests in parallel worker threads.
    - Output dimensionality (MRL) aligns vectors to the vector database collection size.
    - Automatic fallback triggers if the primary Google model fails or exhausts quota.

    Args:
        texts: A single text string or a list of text strings/chunks to embed.
        model: Primary Google embedding model (default: 'gemini-embedding-2').
        dimension: Desired embedding vector dimension (default: 768).
        batch_size: Number of texts per batch slice (default: 32).
        max_concurrency: Number of concurrent batch worker threads (default: 8).
        task_type: Retrieval task type (default: 'RETRIEVAL_DOCUMENT').
        fallback_model: Fallback model if primary model fails (default: 'gemini-embedding-001').
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
        f"(batch_size={effective_batch_size}, dimension={dimension}, model='{model}', rpm={rate_limiter.rpm})"
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
                        genai_client=client,
                        timeout=timeout,
                        rate_limiter=rate_limiter,
                        max_concurrency=max_concurrency,
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
    """Embed a single text string using Google Gemini Embedding 2 with rate limiting."""
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
    """Asynchronously embed texts with Google Gemini Embedding 2, rate limiting, and fallback."""
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
                        f"Failing over to fallback '{fallback_model}'..."
                    )
                    try:
                        fallback_embs = await _aembed_fallback_batch(
                            batch=batch_texts,
                            fallback_model=fallback_model,
                            dimension=dimension,
                            task_type=task_type,
                            genai_client=client,
                            timeout=timeout,
                            rate_limiter=rate_limiter,
                            max_concurrency=max_concurrency,
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
    """Asynchronously embed a single text string using Google Gemini Embedding 2 with rate limiting."""
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

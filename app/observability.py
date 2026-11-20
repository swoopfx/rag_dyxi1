"""Centralized Logfire Observability and Telemetry Framework.

Provides automatic and explicit instrumentation for all classes, methods, and functions
across the application, recording execution timing, arguments, and explicitly marking
whether each call was SUCCESSFUL or FAILED.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import functools
import inspect
import os
import time
from typing import Any, Callable, Dict, Optional, TypeVar, Union

from loguru import logger

# Import settings safely
try:
    from app.config import settings
except ImportError:
    settings = None

# Logfire core imports
try:
    import logfire
except ImportError:
    logfire = None


F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")

SENSITIVE_PARAM_NAMES = {
    "password",
    "passwd",
    "pwd",
    "token",
    "api_key",
    "secret",
    "auth",
    "credentials",
    "bearer",
}


def _sanitize_value(key: str, val: Any) -> Any:
    """Sanitize sensitive parameters from trace payloads."""
    lower_k = key.lower()
    if any(s in lower_k for s in SENSITIVE_PARAM_NAMES):
        return "***REDACTED***"
    if isinstance(val, (str, bytes)) and len(val) > 200:
        return f"<{type(val).__name__} len={len(val)}>"
    if isinstance(val, (list, tuple)) and len(val) > 10:
        return f"<{type(val).__name__} len={len(val)}>"
    if isinstance(val, dict) and len(val) > 10:
        return f"<dict keys={len(val)}>"
    return val


class _NoopSpan:
    """Safe fallback span when Logfire is disabled or unconfigured."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_attributes(self, attributes: Dict[str, Any]) -> None:
        pass

    def record_exception(self, exception: BaseException) -> None:
        pass


_NOOP_SPAN = _NoopSpan()


def setup_observability(service_name: Optional[str] = None) -> bool:
    """Initialize and configure global Logfire observability with auto-instrumentations.

    Enables:
    - Pydantic model automatic instrumentation
    - HTTP requests and HTTPX client instrumentation
    - Loguru log stream forwarding into Logfire spans
    - Auto-tracing across all 'app' modules

    Returns:
        True if Logfire was successfully initialized, False otherwise.
    """
    if logfire is None:
        logger.warning("[Observability] logfire package is not installed.")
        return False

    token = getattr(settings, "LOGFIRE_TOKEN", None) or os.getenv("LOGFIRE_TOKEN")
    svc_name = (
        service_name
        or getattr(settings, "LOGFIRE_INGESTION_SERVICE_NAME", None)
        or os.getenv("LOGFIRE_INGESTION_SERVICE_NAME", "rag-ingestion-service")
    )

    try:
        logfire.configure(
            send_to_logfire="if-token-present",
            token=token,
            service_name=svc_name,
            inspect_arguments=False,
        )

        # 1. Auto-instrument Pydantic models
        try:
            logfire.instrument_pydantic()
        except Exception:
            pass

        # 2. Auto-instrument HTTP clients
        try:
            logfire.instrument_httpx()
        except Exception:
            pass

        try:
            logfire.instrument_requests()
        except Exception:
            pass

        # 3. Forward Loguru logs into active Logfire spans
        try:
            logger.add(**logfire.loguru_handler())
        except Exception:
            pass

        # 4. Install auto-tracing for all non-generator functions in 'app'
        try:
            logfire.install_auto_tracing(
                modules=["app"],
                min_duration=0,
                check_imported_modules="ignore",
            )
        except Exception as e:
            logger.debug(f"[Observability] Auto-tracing registration notice: {e}")

        logger.info(f"[Observability] Logfire initialized successfully for service '{svc_name}'.")
        return True

    except Exception as e:
        logger.warning(f"[Observability] Failed to configure Logfire: {e}")
        return False


def instrument_sqlalchemy_engine(engine: Any) -> None:
    """Instrument an active SQLAlchemy engine with Logfire SQL query tracing."""
    if logfire is not None and engine is not None:
        try:
            logfire.instrument_sqlalchemy(engine)
            logger.debug("[Observability] SQLAlchemy engine instrumented with Logfire.")
        except Exception as e:
            logger.debug(f"[Observability] SQLAlchemy instrumentation notice: {e}")


@contextmanager
def observe_block(name: str, **attributes: Any):
    """Context manager for observing a specific block of code with SUCCESS/FAILED tracking.

    Usage:
        with observe_block("custom_operation", file_name="sample.md") as span:
            perform_task()
    """
    start_time = time.perf_counter()
    if logfire is not None:
        with logfire.span(name, **attributes) as span:
            try:
                yield span
                duration_ms = (time.perf_counter() - start_time) * 1000.0
                span.set_attribute("execution.status", "SUCCESS")
                span.set_attribute("execution.success", True)
                span.set_attribute("execution.duration_ms", round(duration_ms, 2))
                logger.debug(f"[Logfire] [SUCCESS] '{name}' finished in {duration_ms:.2f}ms")
            except Exception as exc:
                duration_ms = (time.perf_counter() - start_time) * 1000.0
                span.set_attribute("execution.status", "FAILED")
                span.set_attribute("execution.success", False)
                span.set_attribute("execution.duration_ms", round(duration_ms, 2))
                span.set_attribute("execution.error_type", type(exc).__name__)
                span.set_attribute("execution.error_message", str(exc))
                span.record_exception(exc)
                logger.error(
                    f"[Logfire] [FAILED] '{name}' failed after {duration_ms:.2f}ms: "
                    f"{type(exc).__name__}: {exc}"
                )
                raise
    else:
        try:
            yield _NOOP_SPAN
        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(f"[FAILED] '{name}' failed after {duration_ms:.2f}ms: {exc}")
            raise


def observe(
    name: Optional[str] = None,
    capture_args: bool = True,
    capture_result: bool = False,
) -> Callable[[Union[F, type]], Union[F, type]]:
    """Universal decorator for observing functions, methods, and entire classes.

    When applied to a CLASS:
        Automatically instruments all public methods on the class.

    When applied to a FUNCTION or METHOD:
        Opens a Logfire span, tracks arguments, measures duration, and explicitly
        marks execution status as 'SUCCESS' or 'FAILED' with error details.

    Usage:
        @observe()
        class DatabaseManager:
            def save(self, data): ...

        @observe("custom_step")
        def process_item(item): ...
    """

    def decorator(target: Union[F, type]) -> Union[F, type]:
        # Handle Class decoration
        if inspect.isclass(target):
            cls = target
            cls_name = name or cls.__name__
            for attr_name, attr_val in list(cls.__dict__.items()):
                # Decorate public functions, staticmethods, and classmethods
                if not attr_name.startswith("__"):
                    if inspect.isfunction(attr_val):
                        method_span_name = f"{cls_name}.{attr_name}"
                        setattr(
                            cls,
                            attr_name,
                            observe(
                                name=method_span_name,
                                capture_args=capture_args,
                                capture_result=capture_result,
                            )(attr_val),
                        )
                    elif isinstance(attr_val, (staticmethod, classmethod)):
                        func = attr_val.__func__
                        method_span_name = f"{cls_name}.{attr_name}"
                        wrapped = observe(
                            name=method_span_name,
                            capture_args=capture_args,
                            capture_result=capture_result,
                        )(func)
                        setattr(
                            cls,
                            attr_name,
                            type(attr_val)(wrapped),
                        )
            return cls

        # Handle Function or Method decoration
        func = target
        span_name = name or getattr(func, "__qualname__", getattr(func, "__name__", "function"))

        # 1. Asynchronous Generator Function
        if inspect.isasyncgenfunction(func):

            @functools.wraps(func)
            async def async_gen_wrapper(*args: Any, **kwargs: Any) -> Any:
                span_attrs: Dict[str, Any] = {}
                if capture_args:
                    try:
                        sig = inspect.signature(func)
                        bound = sig.bind_partial(*args, **kwargs)
                        for k, v in bound.arguments.items():
                            if k != "self" and k != "cls":
                                span_attrs[f"arg.{k}"] = _sanitize_value(k, v)
                    except Exception:
                        pass

                start_time = time.perf_counter()
                if logfire is not None:
                    with logfire.span(span_name, **span_attrs) as span:
                        item_count = 0
                        try:
                            async for item in func(*args, **kwargs):
                                item_count += 1
                                yield item
                            duration_ms = (time.perf_counter() - start_time) * 1000.0
                            span.set_attribute("execution.status", "SUCCESS")
                            span.set_attribute("execution.success", True)
                            span.set_attribute("execution.duration_ms", round(duration_ms, 2))
                            span.set_attribute("result.item_count", item_count)
                            logger.debug(
                                f"[Logfire] [SUCCESS] '{span_name}' yielded {item_count} items in {duration_ms:.2f}ms"
                            )
                        except Exception as exc:
                            duration_ms = (time.perf_counter() - start_time) * 1000.0
                            span.set_attribute("execution.status", "FAILED")
                            span.set_attribute("execution.success", False)
                            span.set_attribute("execution.duration_ms", round(duration_ms, 2))
                            span.set_attribute("execution.error_type", type(exc).__name__)
                            span.set_attribute("execution.error_message", str(exc))
                            span.record_exception(exc)
                            logger.error(
                                f"[Logfire] [FAILED] '{span_name}' failed after {duration_ms:.2f}ms: "
                                f"{type(exc).__name__}: {exc}"
                            )
                            raise
                else:
                    async for item in func(*args, **kwargs):
                        yield item

            return async_gen_wrapper  # type: ignore

        # 2. Synchronous Generator Function
        elif inspect.isgeneratorfunction(func):

            @functools.wraps(func)
            def sync_gen_wrapper(*args: Any, **kwargs: Any) -> Any:
                span_attrs: Dict[str, Any] = {}
                if capture_args:
                    try:
                        sig = inspect.signature(func)
                        bound = sig.bind_partial(*args, **kwargs)
                        for k, v in bound.arguments.items():
                            if k != "self" and k != "cls":
                                span_attrs[f"arg.{k}"] = _sanitize_value(k, v)
                    except Exception:
                        pass

                start_time = time.perf_counter()
                if logfire is not None:
                    with logfire.span(span_name, **span_attrs) as span:
                        item_count = 0
                        try:
                            for item in func(*args, **kwargs):
                                item_count += 1
                                yield item
                            duration_ms = (time.perf_counter() - start_time) * 1000.0
                            span.set_attribute("execution.status", "SUCCESS")
                            span.set_attribute("execution.success", True)
                            span.set_attribute("execution.duration_ms", round(duration_ms, 2))
                            span.set_attribute("result.item_count", item_count)
                            logger.debug(
                                f"[Logfire] [SUCCESS] '{span_name}' yielded {item_count} items in {duration_ms:.2f}ms"
                            )
                        except Exception as exc:
                            duration_ms = (time.perf_counter() - start_time) * 1000.0
                            span.set_attribute("execution.status", "FAILED")
                            span.set_attribute("execution.success", False)
                            span.set_attribute("execution.duration_ms", round(duration_ms, 2))
                            span.set_attribute("execution.error_type", type(exc).__name__)
                            span.set_attribute("execution.error_message", str(exc))
                            span.record_exception(exc)
                            logger.error(
                                f"[Logfire] [FAILED] '{span_name}' failed after {duration_ms:.2f}ms: "
                                f"{type(exc).__name__}: {exc}"
                            )
                            raise
                else:
                    yield from func(*args, **kwargs)

            return sync_gen_wrapper  # type: ignore

        # 3. Asynchronous Coroutine Function
        elif asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                span_attrs: Dict[str, Any] = {}
                if capture_args:
                    try:
                        sig = inspect.signature(func)
                        bound = sig.bind_partial(*args, **kwargs)
                        for k, v in bound.arguments.items():
                            if k != "self" and k != "cls":
                                span_attrs[f"arg.{k}"] = _sanitize_value(k, v)
                    except Exception:
                        pass

                start_time = time.perf_counter()
                if logfire is not None:
                    with logfire.span(span_name, **span_attrs) as span:
                        try:
                            result = await func(*args, **kwargs)
                            duration_ms = (time.perf_counter() - start_time) * 1000.0
                            span.set_attribute("execution.status", "SUCCESS")
                            span.set_attribute("execution.success", True)
                            span.set_attribute("execution.duration_ms", round(duration_ms, 2))
                            if capture_result and result is not None:
                                span.set_attribute("result.summary", _sanitize_value("result", result))
                            logger.debug(
                                f"[Logfire] [SUCCESS] '{span_name}' finished in {duration_ms:.2f}ms"
                            )
                            return result
                        except Exception as exc:
                            duration_ms = (time.perf_counter() - start_time) * 1000.0
                            span.set_attribute("execution.status", "FAILED")
                            span.set_attribute("execution.success", False)
                            span.set_attribute("execution.duration_ms", round(duration_ms, 2))
                            span.set_attribute("execution.error_type", type(exc).__name__)
                            span.set_attribute("execution.error_message", str(exc))
                            span.record_exception(exc)
                            logger.error(
                                f"[Logfire] [FAILED] '{span_name}' failed after {duration_ms:.2f}ms: "
                                f"{type(exc).__name__}: {exc}"
                            )
                            raise
                else:
                    try:
                        return await func(*args, **kwargs)
                    except Exception as exc:
                        duration_ms = (time.perf_counter() - start_time) * 1000.0
                        logger.error(f"[FAILED] '{span_name}' failed after {duration_ms:.2f}ms: {exc}")
                        raise

            return async_wrapper  # type: ignore

        # 4. Standard Synchronous Function or Method
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                span_attrs: Dict[str, Any] = {}
                if capture_args:
                    try:
                        sig = inspect.signature(func)
                        bound = sig.bind_partial(*args, **kwargs)
                        for k, v in bound.arguments.items():
                            if k != "self" and k != "cls":
                                span_attrs[f"arg.{k}"] = _sanitize_value(k, v)
                    except Exception:
                        pass

                start_time = time.perf_counter()
                if logfire is not None:
                    with logfire.span(span_name, **span_attrs) as span:
                        try:
                            result = func(*args, **kwargs)
                            duration_ms = (time.perf_counter() - start_time) * 1000.0
                            span.set_attribute("execution.status", "SUCCESS")
                            span.set_attribute("execution.success", True)
                            span.set_attribute("execution.duration_ms", round(duration_ms, 2))
                            if capture_result and result is not None:
                                span.set_attribute("result.summary", _sanitize_value("result", result))
                            logger.debug(
                                f"[Logfire] [SUCCESS] '{span_name}' finished in {duration_ms:.2f}ms"
                            )
                            return result
                        except Exception as exc:
                            duration_ms = (time.perf_counter() - start_time) * 1000.0
                            span.set_attribute("execution.status", "FAILED")
                            span.set_attribute("execution.success", False)
                            span.set_attribute("execution.duration_ms", round(duration_ms, 2))
                            span.set_attribute("execution.error_type", type(exc).__name__)
                            span.set_attribute("execution.error_message", str(exc))
                            span.record_exception(exc)
                            logger.error(
                                f"[Logfire] [FAILED] '{span_name}' failed after {duration_ms:.2f}ms: "
                                f"{type(exc).__name__}: {exc}"
                            )
                            raise
                else:
                    try:
                        return func(*args, **kwargs)
                    except Exception as exc:
                        duration_ms = (time.perf_counter() - start_time) * 1000.0
                        logger.error(f"[FAILED] '{span_name}' failed after {duration_ms:.2f}ms: {exc}")
                        raise

            return sync_wrapper  # type: ignore

    return decorator


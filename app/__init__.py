"""Enterprise Agentic RAG Application Package."""

import os
import sys
from pathlib import Path

# Auto-route execution to project virtual environment (.venv) if run with external Python
_project_root = Path(__file__).resolve().parent.parent
_venv_python = _project_root / ".venv" / "Scripts" / "python.exe"
if _venv_python.exists() and sys.executable.lower() != str(_venv_python).lower() and not os.environ.get("_RAG_VENV_ACTIVE"):
    import subprocess
    env = os.environ.copy()
    env["_RAG_VENV_ACTIVE"] = "1"
    args = sys.orig_argv[1:] if hasattr(sys, "orig_argv") else sys.argv
    result = subprocess.run([str(_venv_python)] + args, env=env)
    sys.exit(result.returncode)

import contextlib
import functools
from typing import Any, Callable, Dict, Optional, TypeVar

try:
    from app.config import settings
except ImportError:
    settings = None

# Configure Logfire observability across the entire app
try:
    import logfire

    _token = getattr(settings, "LOGFIRE_TOKEN", None) or os.getenv("LOGFIRE_TOKEN")
    _service = (
        getattr(settings, "LOGFIRE_INGESTION_SERVICE_NAME", None)
        or os.getenv("LOGFIRE_INGESTION_SERVICE_NAME", "rag-ingestion-service")
    )
    logfire.configure(
        send_to_logfire="if-token-present",
        token=_token,
        service_name=_service,
    )
except Exception:
    logfire = None


class _NoopSpan:
    """Safe fallback span when Logfire is disabled or unconfigured."""

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
def observe_block(name: str, **attributes: Any):
    """Context manager for observing a block of code directly with Logfire."""
    if logfire is not None:
        try:
            cm = logfire.span(name, **attributes)
        except Exception:
            cm = None

        if cm is not None:
            with cm as span:
                yield span
            return

    yield _NOOP_SPAN


F = TypeVar("F", bound=Callable[..., Any])


def observe(name: Optional[str] = None, **kwargs: Any) -> Callable[[F], F]:
    """Decorator to trace functions directly with Logfire."""
    def decorator(func: F) -> F:
        span_name = name or func.__qualname__

        @functools.wraps(func)
        def wrapper(*args: Any, **func_kwargs: Any) -> Any:
            with observe_block(span_name):
                return func(*args, **func_kwargs)

        return wrapper  # type: ignore

    return decorator


def setup_observability(*args: Any, **kwargs: Any) -> bool:
    """Idempotent Logfire setup helper."""
    return logfire is not None


def instrument_sqlalchemy_engine(engine: Any) -> None:
    """Instrument SQLAlchemy engine with Logfire."""
    if logfire is not None and engine is not None:
        try:
            logfire.instrument_sqlalchemy(engine)
        except Exception:
            pass


__all__ = [
    "logfire",
    "observe",
    "observe_block",
    "setup_observability",
    "instrument_sqlalchemy_engine",
]

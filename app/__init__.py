"""Enterprise Agentic RAG Application Package."""

from app.observability import (
    instrument_sqlalchemy_engine,
    observe,
    observe_block,
    setup_observability,
)

# Automatically configure Logfire observability across the entire app
setup_observability()

__all__ = [
    "observe",
    "observe_block",
    "setup_observability",
    "instrument_sqlalchemy_engine",
]

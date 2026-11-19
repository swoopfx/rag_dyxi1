# Enterprise Agentic Multimodal RAG Engine

A production-grade Retrieval-Augmented Generation (RAG) system with a **Summary-Indexed Multi-Vector Architecture**, combining high-precision semantic search in **Qdrant Vector Database** with exact, uncorrupted structured data persistence in **MySQL**.

---

## Key Highlights

- **Dual-Store Multi-Vector Indexing**: Natural language summaries of structured tables, LaTeX formulas, and visual elements are embedded as dense vectors into Qdrant (`dyxi_rag`), while intact Markdown tables, raw JSON rows, exact LaTeX strings, and image metadata are stored in MySQL.
- **Deterministic Traceability**: Shared deterministic keys (`table_id`, `formula_id`, `image_id`) enable instant $O(1)$ relational hydration when vector search hits match summaries.
- **Passage-to-Table Page Joins**: Automatic co-location joins attach any tables appearing on the same page as a retrieved text chunk.
- **LangChain LCEL Ingestion Pipeline**: Composable runnables chained as `load -> chunk -> embed -> store`.
- **Primary & Fallout Model Support**: Primary embedding via Google Gemini Embedding 2 (`gemini-embedding-2`, 768-dim), with automatic rate-limit failover to Jina AI Multimodal (`jina-embeddings-v4`).
- **Distributed Observability**: Integrated Logfire telemetry under service name `rag-ingestion-service`.
- **Zero-Downtime Resilience**: Automatic failover to local SQLite (`rag_nontextual_metadata.db`) and local persistent Qdrant (`./qdrant_storage`) if external database services are unreachable.

---

## Documentation

Full architectural specifications, data contracts, entity-relationship diagrams, and usage guides are available in the [docs/](file:///c:/llm/RAG/docs) directory:

1. [Architecture & Traceability Guide (Consolidated)](file:///c:/llm/RAG/docs/ARCHITECTURE_AND_TRACEABILITY.md)
   - Architectural design and multi-vector concept
   - Entity-Relationship Diagram (ERD) & schema definitions
   - Deterministic identifier generation formulas
   - Ingestion and query-time retrieval sequence diagrams
   - Modified components and files overview
   - CLI and programmatic usage examples
   - Verified integration test output
2. [Traceability & Queryable Relationship Deep Dive](file:///c:/llm/RAG/docs/TRACEABILITY_AND_QUERY_RETRIEVAL.md)
   - Dedicated specification of the MySQL $\leftrightarrow$ Qdrant dual-store contract
   - Forward retrieval, co-located page joins, and reverse auditing
   - Concrete Python and SQL query examples

---

## Quick Start

### 1. Ingest Documents

```powershell
# Recursive ingestion on DATA/ directory
python -m app.ingestion.processor --data-dir DATA

# Ingestion using fallout model ('jina-embeddings-v4')
python -m app.ingestion.processor --data-dir DATA --fallout

# Single file ingestion
python -m app.ingestion.processor --file DATA/test_quantum_report.md
```

### 2. Query Retrieval & Prompt Augmentation

```python
from app.ingestion.processor import build_augmented_llm_context

context_info = build_augmented_llm_context(
    query="What is the latency and error rate of Q-Titan 2000?",
    top_k=3
)

print(context_info["llm_context_string"])
```

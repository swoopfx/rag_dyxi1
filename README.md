# Enterprise Agentic Multimodal RAG Engine

A production-grade Retrieval-Augmented Generation (RAG) system with a **Summary-Indexed Multi-Vector Architecture**, combining high-precision semantic search in **Qdrant Vector Database** (`dyxi_stag`) with exact, uncorrupted structured data persistence in **MySQL**.

---

## Key Highlights

- **Dual-Store Multi-Vector Indexing**: Natural language summaries of structured tables, LaTeX formulas, and visual elements (with Gemini Flash Vision descriptions) are embedded as dense vectors into Qdrant (**`dyxi_stag`**), while intact Markdown tables, raw JSON rows, exact LaTeX strings, and image metadata are stored in MySQL.
- **Wipe & Re-Install Capability (`--wipe`)**: Atomic purging and recreation of the Qdrant collection and MySQL relational store for clean re-indexing.
- **Subdirectory Targeting (`--sub-dir`)**: Selectively ingest specific subfolders inside `DATA/` (e.g. `DATA/finetunes`, `DATA/noisy`).
- **Volume Throttling (`--max-files`)**: Limit the number of documents processed per batch for staged rollouts and testing.
- **Deterministic Traceability**: Shared deterministic keys (`table_id`, `formula_id`, `image_id`) enable instant $O(1)$ relational hydration when vector search hits match summaries.
- **LangChain LCEL Ingestion Pipeline**: Composable runnables chained as `load -> chunk -> embed -> store`.
- **Primary & Fallout Model Support**: Primary embedding via Google Gemini Embedding 2 (`gemini-embedding-2`, 768-dim), with automatic rate-limit failover to Jina AI Multimodal (`jina-embeddings-v4`).
- **Distributed Observability**: Integrated Logfire telemetry under service name `rag-ingestion-service`.

---

## Documentation

Full architectural specifications, data contracts, entity-relationship diagrams, and usage guides are available in the [docs/](file:///c:/llm/RAG/docs) directory:

1. [Architecture & Traceability Guide (Consolidated)](file:///c:/llm/RAG/docs/ARCHITECTURE_AND_TRACEABILITY.md)
   - Architectural design and multi-vector concept
   - Gemini Flash Vision image description pipeline
   - Storage wiping, targeted subdirectories, and max-files controls
   - Entity-Relationship Diagram (ERD) & schema definitions
   - Ingestion and query-time retrieval sequence diagrams
   - CLI and programmatic usage examples
2. [Traceability & Queryable Relationship Deep Dive](file:///c:/llm/RAG/docs/TRACEABILITY_AND_QUERY_RETRIEVAL.md)
   - Dedicated specification of the MySQL $\leftrightarrow$ Qdrant (`dyxi_stag`) dual-store contract
   - Forward retrieval, co-located page joins, and reverse auditing
   - Concrete Python and SQL query examples

---

## Quick Start CLI

```bash
# 0. Initialize MySQL database (rag_metadata) and lookup tables
./scripts/setup_database.sh                         # Bash (Linux/macOS/Git Bash)
.\scripts\setup_database.ps1                       # PowerShell (Windows)

# 1. Full clean wipe and re-install of DATA/ into dyxi_stag
python -m app.ingestion.processor --wipe --collection dyxi_stag

# 2. Ingest only files inside DATA/finetunes
python -m app.ingestion.processor --sub-dir finetunes

# 3. Clean wipe and ingest at most 5 files from DATA/finetunes
python -m app.ingestion.processor --wipe --sub-dir finetunes --max-files 5

# 4. Ingest with fallout model ('jina-embeddings-v4')
python -m app.ingestion.processor --sub-dir noisy --fallout
```

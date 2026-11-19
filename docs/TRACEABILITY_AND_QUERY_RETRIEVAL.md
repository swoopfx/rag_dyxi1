# Traceability & Queryable Relationship Guide: MySQL <-> Qdrant

This document details the exact **traceable and queryable relationship** between the structured artifacts (tables, images, LaTeX formulas) stored in MySQL and the dense vector embeddings stored in Qdrant Vector Database, including the **Gemini Flash Vision Image Description** pipeline.

---

## 1. Architectural Summary: The Dual-Store Contract

In this enterprise RAG implementation:
- **Qdrant Vector Database** stores semantic summary embeddings ($768$-dimensional vectors) optimized for high-recall Cosine similarity search across text passages, table schemas, formulas, and AI visual descriptions.
- **MySQL Database** stores pristine, intact source artifacts (raw Markdown tables, JSON row arrays, exact LaTeX math formulas, high-resolution visual assets, and AI image descriptions).

```mermaid
sequenceDiagram
    autonumber
    actor User as User Question
    participant Ret as retrieval.py (processor)
    participant Qd as Qdrant Vector DB
    participant My as MySQL (Indexed Lookup)
    participant LLM as Target LLM

    User->>Ret: "Query Engine and Vector Index architecture"
    Ret->>Qd: Cosine Search(embed_query(q), limit=5)
    Qd-->>Ret: Hit: content_type="image_summary", image_id="b885...#image_000", score=0.8044
    Ret->>My: SELECT description, saved_path, base64_uri FROM rag_extracted_images WHERE image_id="b885...#image_000"
    My-->>Ret: AI Visual Description, image path, base64 URI
    Ret->>Ret: Build formatted prompt block ### [IMAGE ELEMENT] ...
    Ret->>LLM: Ingest pristine context + User Query
    LLM-->>User: Accurate answer with visual insights & architectural relationships
```

---

## 2. Shared Foreign Key & Identifier Contracts

Every entity possesses a deterministic identifier shared between MySQL and Qdrant:

| Entity Type | Shared Foreign Key | Formula | MySQL Table & Column | Qdrant Point Payload |
|---|---|---|---|---|
| **Document** | `document_id` | `uuid.uuid5(NAMESPACE_URL, file_path)` | `rag_document_metadata.document_id` (Unique Index) | `payload["document_id"]` |
| **Table** | `table_id` | `{document_id}#table_{index:03d}` | `rag_extracted_tables.table_id` (Unique Index) | `payload["table_id"]` |
| **LaTeX Formula** | `formula_id` | `{document_id}#formula_{index:03d}` | `rag_extracted_formulas.formula_id` (Unique Index) | `payload["formula_id"]` |
| **Image** | `image_id` | `{document_id}#image_{index:03d}` | `rag_extracted_images.image_id` (Unique Index) | `payload["image_id"]` |
| **Text Chunk** | `chunk_id` | `{file_name}#chunk_{index:05d}` | Referenceable via `file_name` & `page_no` | `payload["chunk_id"]` |

---

## 3. Gemini Flash Vision Image Description Pipeline

When documents contain diagrams, charts, screenshots, or uncaptioned photos:

1. **Extraction (`describe_image_with_gemini`)**:
   - Uses `gemini-flash-latest` (with automatic failover to `gemini-3.1-flash-lite`).
   - Analyzes raw image bytes and generates a 2-3 sentence semantic summary describing:
     * Category (flowchart, architecture diagram, UI mockup, data chart, photo).
     * Text, numbers, component boxes, and labels visible in the image.
     * Architectural relationships and data flows.
2. **Qdrant Vector Embedding**:
   - The visual description is included in `image_summary` and vectorized into Qdrant (`dyxi_rag`).
   - `payload["description"]` is stored with the point for immediate query inspection.
3. **MySQL Relational Persistence**:
   - The description is saved to `rag_extracted_images.description` alongside the disk path, width, height, and base64 string.
4. **Query-Time Retrieval**:
   - Vector search matches user questions against the AI visual description.
   - Retrieval queries MySQL via `image_id` and attaches the description to the prompt context.

---

## 4. Query Hydration Flow at Inference Time

Implemented in [`retrieve_similar_chunks()`](file:///c:/llm/RAG/app/ingestion/processor.py#L1038):

### Step 1: Semantic Vector Matching
The user query is embedded into an asymmetric search vector using `embed_query(task_type="RETRIEVAL_QUERY")` and queried against Qdrant collection `dyxi_rag`.

### Step 2: Content-Type & Key Inspection
For each matched point in Qdrant:
- `payload["content_type"]` indicates whether a table summary, formula summary, image summary, or text passage matched.
- The shared key (`table_id`, `formula_id`, or `image_id`) is extracted.

### Step 3: O(1) MySQL Indexed Point Lookup

```python
# From app/ingestion/processor.py:

# Case A: Table Summary Matched
if content_type == "table_summary" or table_id:
    target_table_id = table_id or payload.get("chunk_id")
    table_record = db_mgr.get_table_by_id(target_table_id)
    if table_record:
        result_item["intact_table_markdown"] = table_record["markdown"]
        result_item["intact_table_data_rows"] = table_record["data_rows"]
        result_item["formatted_context_for_llm"] = (
            f"### [INTACT TABLE] {target_table_id}\n"
            f"{table_record['markdown']}\n"
        )

# Case B: LaTeX Formula Summary Matched
elif content_type == "formula_summary" or formula_id:
    target_formula_id = formula_id or payload.get("chunk_id")
    formula_record = db_mgr.get_formula_by_id(target_formula_id)
    if formula_record:
        result_item["latex"] = formula_record["latex"]
        result_item["formatted_context_for_llm"] = (
            f"### [LATEX FORMULA] {target_formula_id}\n"
            f"{formula_record['latex']}\n"
        )

# Case C: Image Summary Matched (with Gemini Flash Vision Description)
elif content_type == "image_summary" or image_id:
    target_image_id = image_id or payload.get("chunk_id")
    image_record = db_mgr.get_image_by_id(target_image_id)
    if image_record:
        vis_desc = image_record.get("description") or image_record.get("summary", "")
        result_item["formatted_context_for_llm"] = (
            f"### [IMAGE ELEMENT] {target_image_id} ({image_record['label']})\n"
            f"Caption: {image_record.get('caption', 'None')}\n"
            f"Visual Description: {vis_desc}\n"
            f"Summary: {image_record.get('summary', '')}\n"
        )
```

---

## 5. Multi-Modal Co-Located Page Joins

When Qdrant matches a **text passage** rather than a table summary, the system automatically checks MySQL for any tables co-located on the same document page:

```python
# Lines 1209-1230 in app/ingestion/processor.py:
related_tables = []
if source_file and page_numbers:
    doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(Path(source_file).resolve())))
    with db_mgr.get_session() as session:
        tbl_records = (
            session.query(ExtractedTableRecord)
            .filter(
                ExtractedTableRecord.document_id == doc_id,
                ExtractedTableRecord.page_no.in_(page_numbers),
            )
            .all()
        )
        related_tables = [
            {"table_id": t.table_id, "caption": t.caption, "markdown": t.markdown}
            for t in tbl_records
        ]
result_item["related_tables"] = related_tables
```

---

## 5.1 Text Chunking Architecture & Provenance (`HybridChunker`)

Text passages are chunked using Docling's `HybridChunker` with structure-aware contextualization:

1. **Heading Breadcrumb Injection**: Prepends hierarchical document section paths (e.g. `System Architecture > Storage Layer`) into `contextualized_text`. This resolves ambiguous sub-clauses for high-recall vector search in Qdrant without adding costly LLM summarization calls.
2. **Peer Node Merging (`merge_peers=True`)**: Aggregates undersized bullet points and list items under the same parent section up to `max_tokens=1200` to eliminate microscopic, contextless chunks.
3. **Zero-Dependency `GeminiTokenEstimator`**: Uses an in-memory character/word heuristic tailored for Gemini Embedding 2 (~4 chars/token), completely avoiding external HuggingFace/OpenAI tokenizer downloads and network delays.
4. **Provenance Tracking**: Every chunk retains its originating `page_numbers`, enabling the exact co-located table joins shown in Section 5.

---

## 6. Reverse Auditing: Querying Qdrant from MySQL

Given any `image_id`, `table_id`, or `formula_id` in MySQL, you can locate its corresponding vector embedding in Qdrant:

```python
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.config import settings

client = QdrantClient(url=settings.QDRANT_CLUSTER_ENDPOINT, api_key=settings.QDRANT_API_KEY, prefer_grpc=False)

# Look up the vector point for a specific image_id
points, _ = client.scroll(
    collection_name="dyxi_rag",
    scroll_filter=models.Filter(
        must=[models.FieldCondition(key="image_id", match=models.MatchValue(value="b88513c1-33d7-5777-aa86-a1f9a25c0db4#image_000"))]
    ),
    limit=1,
    with_payload=True
)

if points:
    pt = points[0]
    print(f"Vector Point ID: {pt.id}")
    print(f"Visual Description: {pt.payload.get('description')}")
    print(f"Summary Embedded: {pt.payload.get('summary')}")
```

---

## 7. Real-World Verified Test Logs

```text
================================================================================
TEST EXECUTION LOGS (TABLES, FORMULAS, AND VISION IMAGES)
================================================================================

[1] Image Ingestion & Flash Vision Description:
    File: DATA/test_system_arch.png
    Document ID: b88513c1-33d7-5777-aa86-a1f9a25c0db4
    Gemini Flash Vision Output:
    "This diagram illustrates a 'Transformer Microservice Architecture' featuring two primary
    components connected by a direct line. The chart displays a 'Query Engine' on the left and
    a 'Vector Index' on the right, representing a standard architectural relationship between
    a retrieval interface and its underlying data store."
    MySQL Stored: 1 image record with description in rag_extracted_images
    Qdrant Stored: 1 vector embedding with visual description payload in dyxi_rag

[2] Retrieval for Image Query: "Query Engine and Vector Index architecture"
    Hit #1: type=image_summary, score=0.8044
    Extracted Key: image_id=b88513c1-33d7-5777-aa86-a1f9a25c0db4#image_000
    MySQL Hydration Result:
    Visual Description: "This image is a simple flowchart depicting a connection between two
    primary architectural components. The diagram shows a 'Query Engine' connected by a line
    to a 'Vector Index.'"

[3] Retrieval for Table Query: "What is the latency and error rate of Q-Titan 2000?"
    Hit #1: type=table_summary, score=0.6974
    MySQL Hydrated Intact Markdown Table:
    | Processor Model   |   Qubits Count |   Quantum Volume |   Error Rate (%) |   Latency (ms) |
    |-------------------|----------------|------------------|------------------|----------------|
    | Q-Titan 2000      |           2048 |            65536 |           0.0012 |           1.45 |

[4] Retrieval for Formula Query: "relativistic energy mass formula"
    Hit #1: type=formula_summary, score=0.7241
    MySQL Hydrated Exact LaTeX:
    $$E = mc^2 + \frac{p^2}{2m}$$
```

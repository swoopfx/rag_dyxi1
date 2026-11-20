import os
import urllib.parse
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # --- 1. API Keys & LLM Gateways ---
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_FALLBACK_API_KEY = os.getenv("GROQ_FALLBACK_API_KEY")
    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
    JINA_API_KEY = os.getenv("JINA_API_KEY") or os.getenv("JINA_AI_API_TOKEN")
    BACKEND_URL = os.getenv("BACKEND_URL")
    JUDGE_GROQ = os.getenv("JUDGE_GROQ")

    # --- 2. Qdrant Vector DB Settings ---
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_CLUSTER_ENDPOINT = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "dyxi_stag")
    QDRANT_LOCAL_PATH = os.getenv("QDRANT_LOCAL_PATH", "./qdrant_storage")
    QDRANT_TIMEOUT_SECONDS = float(os.getenv("QDRANT_TIMEOUT_SECONDS", "30.0"))
    QDRANT_DISTANCE_METRIC = os.getenv("QDRANT_DISTANCE_METRIC", "Cosine")

    # --- 3. MySQL Relational DB Settings ---
    MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
    MYSQL_USER = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "Oluwaseun1@")
    MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "rag_metadata")
    MYSQL_URL = os.getenv("MYSQL_URL")  # Full override if provided
    SQLITE_FALLBACK_URL = os.getenv("SQLITE_FALLBACK_URL", "sqlite:///rag_nontextual_metadata.db")

    # --- 4. Relational Database Table Names (Constant Configuration) ---
    TABLE_NAME_DOCUMENTS = os.getenv("TABLE_NAME_DOCUMENTS", "rag_document_metadata")
    TABLE_NAME_TABLES = os.getenv("TABLE_NAME_TABLES", "rag_extracted_tables")
    TABLE_NAME_IMAGES = os.getenv("TABLE_NAME_IMAGES", "rag_extracted_images")
    TABLE_NAME_FORMULAS = os.getenv("TABLE_NAME_FORMULAS", "rag_extracted_formulas")

    # --- 5. Embedding & Model Configuration ---
    PRIMARY_EMBEDDING_MODEL = os.getenv("PRIMARY_EMBEDDING_MODEL", "gemini-embedding-2")
    FALLBACK_EMBEDDING_MODEL = os.getenv("FALLBACK_EMBEDDING_MODEL", "jina-embeddings-v4")
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))
    EMBEDDING_RPM_LIMIT = int(os.getenv("EMBEDDING_RPM_LIMIT", "600"))
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    USE_FALLOUT_MODEL = os.getenv("USE_FALLOUT_MODEL", "false").lower() in ("true", "1", "yes")

    # --- 6. Vision & Multimodal Model Configuration ---
    GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-flash-latest")
    FALLBACK_VISION_MODEL = os.getenv("FALLBACK_VISION_MODEL", "gemini-3.1-flash-lite")
    ENABLE_VISION_IMAGE_DESCRIPTION = os.getenv("ENABLE_VISION_IMAGE_DESCRIPTION", "true").lower() in ("true", "1", "yes")
    VISION_MAX_OUTPUT_TOKENS = int(os.getenv("VISION_MAX_OUTPUT_TOKENS", "150"))
    VISION_TEMPERATURE = float(os.getenv("VISION_TEMPERATURE", "0.2"))

    # --- 7. Document Chunking & Ingestion Constants ---
    DATA_DIR = os.getenv("DATA_DIR", "DATA")
    SUPPORTED_FILE_EXTENSIONS = [
        ext.strip().lower()
        for ext in os.getenv(
            "SUPPORTED_FILE_EXTENSIONS",
            ".pdf,.docx,.doc,.pptx,.ppt,.xlsx,.xls,.csv,.html,.htm,.md,.markdown,.mdown,.mkd,.txt,.text,.json,.png,.jpg,.jpeg,.tiff,.tif,.bmp,.webp,.gif",
        ).split(",")
        if ext.strip()
    ]
    CHUNK_MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", "1200"))
    CHUNK_MERGE_PEERS = os.getenv("CHUNK_MERGE_PEERS", "true").lower() in ("true", "1", "yes")
    PDF_PAGE_BATCH_SIZE = int(os.getenv("PDF_PAGE_BATCH_SIZE", "30"))
    RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))

    # --- 8. Logfire & Observability ---
    LOGFIRE_TOKEN = os.getenv("LOGFIRE_TOKEN")
    LOGFIRE_INGESTION_SERVICE_NAME = os.getenv("LOGFIRE_INGESTION_SERVICE_NAME", "rag-ingestion-service")
    LOGFIRE_EMBEDDINGS_SERVICE_NAME = os.getenv("LOGFIRE_EMBEDDINGS_SERVICE_NAME", "rag-embeddings-service")
    LOGFIRE_TRACING_ENABLED = os.getenv("LOGFIRE_TRACING_ENABLED", "true").lower() in ("true", "1", "yes")

    LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "true").lower() in ("true", "1", "yes")
    LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "")

    def get_mysql_url(self) -> str:
        """Construct a standardized SQLAlchemy MySQL connection URL with pymysql."""
        if self.MYSQL_URL:
            return self.MYSQL_URL
        user = urllib.parse.quote_plus(self.MYSQL_USER) if self.MYSQL_USER else "root"
        password = (
            f":{urllib.parse.quote_plus(self.MYSQL_PASSWORD)}"
            if self.MYSQL_PASSWORD
            else ""
        )
        host = self.MYSQL_HOST or "localhost"
        port = self.MYSQL_PORT or 3306
        db = self.MYSQL_DATABASE or "rag_metadata"
        return f"mysql+pymysql://{user}{password}@{host}:{port}/{db}?charset=utf8mb4"


settings = Settings()

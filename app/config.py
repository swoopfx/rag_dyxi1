import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # --- API Keys ---
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    JINA_API_KEY = os.getenv("JINA_API_KEY") or os.getenv("JINA_AI_API_TOKEN")

    # --- Qdrant Vector DB Settings ---
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_CLUSTER_ENDPOINT = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "dyxi_stag")

    # --- MySQL Relational DB Settings (for nontextual tables/images) ---
    MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
    MYSQL_USER = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
    MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "rag_metadata")
    MYSQL_URL = os.getenv("MYSQL_URL")  # Full override if provided

    # SQLite fallback path if MySQL server is unreachable
    SQLITE_FALLBACK_URL = os.getenv("SQLITE_FALLBACK_URL", "sqlite:///rag_nontextual_metadata.db")

    # --- Embedding & Model Configuration ---
    PRIMARY_EMBEDDING_MODEL = os.getenv("PRIMARY_EMBEDDING_MODEL", "gemini-embedding-2")
    FALLBACK_EMBEDDING_MODEL = os.getenv("FALLBACK_EMBEDDING_MODEL", "jina-embeddings-v4")
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))
    EMBEDDING_RPM_LIMIT = int(os.getenv("EMBEDDING_RPM_LIMIT", "600"))

    # Flag to prioritize the fallback / fallout model for both ingestion and retrieval
    USE_FALLOUT_MODEL = os.getenv("USE_FALLOUT_MODEL", "false").lower() in ("true", "1", "yes")

    # --- Vision & Multimodal Model Configuration ---
    GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-flash-latest")
    FALLBACK_VISION_MODEL = os.getenv("FALLBACK_VISION_MODEL", "gemini-3.1-flash-lite")
    ENABLE_VISION_IMAGE_DESCRIPTION = os.getenv("ENABLE_VISION_IMAGE_DESCRIPTION", "true").lower() in ("true", "1", "yes")

    # --- Logfire Observability ---
    LOGFIRE_TOKEN = os.getenv("LOGFIRE_TOKEN")
    LOGFIRE_INGESTION_SERVICE_NAME = os.getenv(
        "LOGFIRE_INGESTION_SERVICE_NAME", "rag-ingestion-service"
    )

    # --- Data Paths ---
    DATA_DIR = os.getenv("DATA_DIR", "DATA")

    def get_mysql_url(self) -> str:
        """Construct a standardized SQLAlchemy MySQL connection URL with pymysql."""
        if self.MYSQL_URL:
            return self.MYSQL_URL
        user = self.MYSQL_USER or "root"
        password = f":{self.MYSQL_PASSWORD}" if self.MYSQL_PASSWORD else ""
        host = self.MYSQL_HOST or "localhost"
        port = self.MYSQL_PORT or 3306
        db = self.MYSQL_DATABASE or "rag_metadata"
        return f"mysql+pymysql://{user}{password}@{host}:{port}/{db}?charset=utf8mb4"


settings = Settings()

import os
from dotenv import load_dotenv
load_dotenv()


class Settings:
     GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
     OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
     QDRANT_API_KEY =os.getenv("QDRANT_API_KEY")
     QDRANT_CLUSTER_ENDPOINT=os.getenv("QDRANT_CLUSTER_ENDPOINT")
     QDRANT_COLLECTION ="dyxi_rag"
     EMBEDDING_RPM_LIMIT = int(os.getenv("EMBEDDING_RPM_LIMIT", "600"))


settings = Settings()
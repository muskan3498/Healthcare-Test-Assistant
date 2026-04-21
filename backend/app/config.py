from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
RAG_DIR = DATA_DIR / "rag"
QDRANT_DIR = RAG_DIR / "qdrant"


class Settings(BaseSettings):
    openai_api_key: str | None = None
    openai_model: str = "azure_ai/genailab-maas-DeepSeek-V3-0324"
    openai_base_url: str | None = "https://genailab.tcs.in"
    openai_verify_ssl: bool = False
    openai_timeout_seconds: float = 60.0
    max_upload_size_bytes: int = 10 * 1024 * 1024
    chunk_size: int = 900
    chunk_overlap: int = 150
    retrieval_limit: int = 5
    rag_vector_store_path: str = str(QDRANT_DIR)
    rag_collection_name: str = "nutrition_knowledge"
    rag_embedding_model: str = "azure/genailab-maas-text-embedding-3-large"
    rag_embedding_dimensions: int = 3072
    rag_candidate_k: int = 24
    rag_rerank_top_n: int = 12
    rag_enable_hybrid: bool = True
    rag_enable_rerank: bool = True
    rag_dense_weight: float = 0.62
    rag_lexical_weight: float = 0.38
    rag_indexing_batch_size: int = 24
    rag_allow_local_embedding_fallback: bool = True
    cors_origins: list[str] = ["http://localhost:5173"]

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RAG_DIR.mkdir(parents=True, exist_ok=True)
QDRANT_DIR.mkdir(parents=True, exist_ok=True)

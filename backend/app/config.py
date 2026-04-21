from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"


class Settings(BaseSettings):
    openai_api_key: str | None = None
    openai_model: str = "gpt-4.1-mini"
    max_upload_size_bytes: int = 10 * 1024 * 1024
    chunk_size: int = 900
    chunk_overlap: int = 150
    retrieval_limit: int = 5
    cors_origins: list[str] = ["http://localhost:5173"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

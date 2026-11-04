"""Environment-backed settings for optional retrieval enhancements."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class RetrievalSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RETRIEVAL_",
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    enable_reranker: bool = False
    enable_query_rewriting: bool = False
    enable_query_classification: bool = False
    enable_hyde: bool = False
    bm25_remove_stop_words: bool = True
    bm25_stop_words: list[str] | None = None
    llm_provider: Literal["openai", "genai"] = "genai"
    llm_model: str = "gemini-3.5-flash-lite"
    query_classifier_model: str = "gemini-3.5-flash-lite"
    llm_api_key: SecretStr | None = None
    llm_base_url: str | None = None
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_batch_size: int = Field(default=16, gt=0)
    reranker_max_length: int = Field(default=512, gt=0)
    reranker_timeout_seconds: float = Field(default=2.0, gt=0)
    rerank_candidate_count: int = Field(default=40, gt=0)
    max_rewrites: int = Field(default=2, ge=0, le=2)
    generation_max_tokens: int = Field(default=256, gt=0)
    generation_timeout_seconds: float = Field(default=25.0, gt=0)
    augmentation_cache_ttl_seconds: float = Field(default=900.0, gt=0)
    augmentation_cache_size: int = Field(default=1000, gt=0)


@lru_cache(maxsize=1)
def get_retrieval_settings() -> RetrievalSettings:
    return RetrievalSettings()

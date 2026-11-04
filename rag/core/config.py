"""Cấu hình dùng chung, đọc từ biến môi trường / file .env."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class VlmProviderSettings(BaseModel):
    model: str
    api_key: str = ""
    base_url: str = ""
    max_tokens: int = 1024
    temperature: float = 0.0
    prompt: str = (
        "Extract all readable text, tables, numbers, and structured data from this image. "
        "If the image has no extractable data, return a concise description of the image."
    )


class OpenAISettings(VlmProviderSettings):
    model: str = "gpt-4o-mini"


class GenAISettings(VlmProviderSettings):
    model: str = "gemini-3.6-flash"


class VlmSettings(BaseModel):
    provider: Literal["openai", "genai"] = "genai"
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    genai: GenAISettings = Field(default_factory=GenAISettings)

    @property
    def active(self) -> VlmProviderSettings:
        return self.genai if self.provider == "genai" else self.openai


class StorageSettings(BaseModel):
    postgres_url: SecretStr | None = None
    published_versions_manifest: str = "published_versions.json"
    elasticsearch_url: str = "http://localhost:9200"
    elasticsearch_index: str = "rag-chunks"
    elasticsearch_api_key: SecretStr | None = None
    elasticsearch_username: SecretStr | None = None
    elasticsearch_password: SecretStr | None = None
    elasticsearch_request_timeout: float = 30.0

    @property
    def postgres_url_value(self) -> str:
        if self.postgres_url is None:
            raise ValueError("PostgreSQL URL is not configured; set POSTGRESQL_URL in .env")
        return self.postgres_url.get_secret_value()


class GraphSettings(BaseModel):
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: SecretStr = SecretStr("")
    model: str = "openai/gpt-oss-120b"
    classifier_model: str = "qwen/qwen3.8-27b"
    aliases: dict[str, str] = Field(default_factory=dict)
    keyword_density_threshold: float = Field(default=0.015, ge=0.0, le=1.0)
    minimum_keyword_hits: int = Field(default=1, ge=1)
    rule_link_threshold: float = Field(default=0.82, ge=0.0, le=1.0)
    max_output_tokens: int = Field(default=2048, gt=0)
    timeout_seconds: float = Field(default=30.0, gt=0.0)


class IngestionSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        env_nested_delimiter="__",
    )

    vlm_settings: VlmSettings = Field(default_factory=VlmSettings)
    storage_settings: StorageSettings = Field(default_factory=StorageSettings)
    graph_settings: GraphSettings = Field(default_factory=GraphSettings)
    postgresql_url_env: SecretStr | None = Field(
        default=None,
        validation_alias="POSTGRESQL_URL",
        exclude=True,
    )
    groq_api_key: SecretStr = Field(
        default=SecretStr(""),
        validation_alias="GROQ_API_KEY",
    )

    request_timeout_seconds: float = Field(
        default=30.0,
        validation_alias=AliasChoices("INGESTION_TIMEOUT_SECONDS", "REQUEST_TIMEOUT_SECONDS"),
    )
    max_file_size_mb: int = Field(
        default=50,
        validation_alias=AliasChoices("INGESTION_MAX_FILE_SIZE_MB", "MAX_FILE_SIZE_MB"),
    )

    allowed_file_extensions: tuple[str, ...] = (
        ".pdf",
        ".doc",
        ".docx",
        ".md",
        ".markdown",
        ".html",
        ".htm",
    )

    @model_validator(mode="after")
    def apply_postgresql_url_env(self) -> "IngestionSettings":
        # Support the flat POSTGRESQL_URL key while keeping the URL under storage_settings.
        # The nested STORAGE_SETTINGS__POSTGRES_URL key, when present, takes precedence.
        if self.storage_settings.postgres_url is None and self.postgresql_url_env is not None:
            self.storage_settings.postgres_url = self.postgresql_url_env
        return self

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> IngestionSettings:
    return IngestionSettings()

"""Cấu hình ingestion, đọc từ biến môi trường / file .env."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field
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


class IngestionSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        env_nested_delimiter="__",
    )

    vlm_settings: VlmSettings = Field(default_factory=VlmSettings)

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

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> IngestionSettings:
    return IngestionSettings()

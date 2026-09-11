"""Cấu hình ingestion, đọc từ biến môi trường / file .env."""

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class IngestionSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    vlm_model: str = "gpt-4o-mini"
    vlm_max_tokens: int = 1024
    vlm_temperature: float = 0.0
    vlm_prompt: str = (
        "Extract all readable text, tables, numbers, and structured data from this image. "
        "If the image has no extractable data, return a concise description of the image."
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

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> IngestionSettings:
    return IngestionSettings()

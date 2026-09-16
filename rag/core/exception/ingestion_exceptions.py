"""Lỗi chuẩn hóa cho pipeline ingestion."""


class IngestionError(Exception):
    """Lỗi gốc của module ingestion."""

    def __init__(self, message: str, process: str | None = None, metrics: dict | None = None) -> None:
        super().__init__(message)
        self.process = process
        self.metrics = metrics or {}


class ValidationError(IngestionError):
    """Nguồn URL/file không hợp lệ."""


class ParseError(IngestionError):
    """Docling không đọc được tài liệu."""


class VlmError(IngestionError):
    """Gọi VLM thất bại."""

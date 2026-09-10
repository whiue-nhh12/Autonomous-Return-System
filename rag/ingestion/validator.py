"""Bước VALIDATION: kiểm tra và lấy tài liệu từ URL hoặc file local."""

from __future__ import annotations

import mimetypes
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from urllib.parse import urlparse

import httpx

from rag.ingestion.base import BaseIngestion
from rag.ingestion.exceptions import ValidationError

_URL_SCHEMES = {"http", "https"}
_CONTENT_TYPE_TO_SUFFIX = {
    "application/pdf": ".pdf",
    "application/msword": ".doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "text/markdown": ".md",
    "text/x-markdown": ".md",
    "text/html": ".html",
    "application/xhtml+xml": ".html",
}


@dataclass(slots=True)
class ValidatedSource:
    origin: str
    local_path: Path
    file_type: str
    size_bytes: int
    is_url: bool
    cleanup_path: Path | None = None


class SourceValidator(BaseIngestion):
    process = "VALIDATION"

    def run(self, source: str) -> ValidatedSource:
        started = perf_counter()
        try:
            origin = (source or "").strip()
            if not origin:
                self.fail("Nguồn tài liệu trống.", {"size_bytes": 0}, ValidationError)


            result = self._validate_url(origin) if self._is_url(origin) else self._validate_file(origin)
            self.emit(
                "SUCCESS",
                "Nguồn tài liệu hợp lệ.",
                self.timed_metrics(
                    started,
                    is_url=result.is_url,
                    file_type=result.file_type,
                    size_bytes=result.size_bytes,
                    source_length=len(origin),
                    timeout_seconds=self.settings.request_timeout_seconds,
                    max_file_size_mb=self.settings.max_file_size_mb,
                ),
            )
            return result
        except ValidationError:
            raise
        except Exception as exc:
            self.fail(f"Lỗi không mong đợi khi kiểm tra nguồn: {exc}", self.timed_metrics(started), ValidationError)

    @staticmethod
    def _is_url(source: str) -> bool:
        parsed = urlparse(source)
        return parsed.scheme.lower() in _URL_SCHEMES and bool(parsed.netloc)

    def _validate_file(self, source: str) -> ValidatedSource:
        path = Path(source).expanduser().resolve()
        if not path.exists() or not path.is_file():
            self.fail(f"Đường dẫn không tồn tại hoặc không phải file: {path}", {"path": str(path)}, ValidationError)

        suffix = path.suffix.lower()
        if suffix not in self.settings.allowed_file_extensions:
            self.fail(
                f"Định dạng {suffix or '(không có)'} không được phép. Cho phép: {', '.join(self.settings.allowed_file_extensions)}.",
                {"file_type": suffix, "allowed": list(self.settings.allowed_file_extensions)},
                ValidationError,
            )

        size_bytes = path.stat().st_size
        self._assert_size(size_bytes)
        return ValidatedSource(
            origin=str(path),
            local_path=path,
            file_type=suffix.lstrip("."),
            size_bytes=size_bytes,
            is_url=False,
        )

    def _validate_url(self, url: str) -> ValidatedSource:
        parsed = urlparse(url)
        if parsed.scheme.lower() not in _URL_SCHEMES or not parsed.netloc:
            self.fail(f"URL không hợp lệ: {url}", {"url": url}, ValidationError)

        timeout = httpx.Timeout(self.settings.request_timeout_seconds)
        tmp_path: Path | None = None
        try:
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                with client.stream("GET", url) as response:
                    response.raise_for_status()
                    header_length = response.headers.get("content-length")
                    if header_length:
                        self._assert_size(int(header_length))

                    suffix = self._suffix_from_url(url, response.headers.get("content-type", ""))
                    if suffix not in self.settings.allowed_file_extensions:
                        self.fail(
                            f"URL không trỏ tới định dạng được phép ({suffix or 'unknown'}).",
                            {"file_type": suffix, "content_type": response.headers.get("content-type")},
                            ValidationError,
                        )

                    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                    tmp_path = Path(tmp.name)
                    size_bytes = 0
                    try:
                        for chunk in response.iter_bytes(64 * 1024):
                            size_bytes += len(chunk)
                            if size_bytes > self.settings.max_file_size_bytes:
                                self.fail(
                                    f"Kích thước tải về vượt {self.settings.max_file_size_mb}MB.",
                                    {"size_bytes": size_bytes, "max_file_size_bytes": self.settings.max_file_size_bytes},
                                    ValidationError,
                                )
                            tmp.write(chunk)
                    finally:
                        tmp.close()
        except ValidationError:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
            raise
        except httpx.TimeoutException as exc:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
            self.fail(
                f"Timeout khi tải URL sau {self.settings.request_timeout_seconds}s.",
                {"timeout_seconds": self.settings.request_timeout_seconds, "error": str(exc)},
                ValidationError,
            )
        except httpx.HTTPError as exc:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
            self.fail(f"Không tải được URL: {exc}", {"url": url}, ValidationError)

        path = tmp_path
        return ValidatedSource(
            origin=url,
            local_path=path,
            file_type=suffix.lstrip("."),
            size_bytes=size_bytes,
            is_url=True,
            cleanup_path=path,
        )

    def _assert_size(self, size_bytes: int) -> None:
        if size_bytes > self.settings.max_file_size_bytes:
            self.fail(
                f"Kích thước {size_bytes} bytes vượt giới hạn {self.settings.max_file_size_mb}MB.",
                {"size_bytes": size_bytes, "max_file_size_bytes": self.settings.max_file_size_bytes},
                ValidationError,
            )

    @staticmethod
    def _suffix_from_url(url: str, content_type: str) -> str:
        mime = content_type.split(";")[0].strip().lower()
        if mime in _CONTENT_TYPE_TO_SUFFIX:
            return _CONTENT_TYPE_TO_SUFFIX[mime]
        guessed = Path(urlparse(url).path).suffix.lower()
        if guessed:
            return guessed
        ext = mimetypes.guess_extension(mime) or ""
        return ext

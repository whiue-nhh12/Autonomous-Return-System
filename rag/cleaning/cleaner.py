"""Lọc block và làm sạch text cho bước embedding.

Module này chỉ nhận kết quả của ingestion; nó không được gọi từ ingestion pipeline.
"""

from __future__ import annotations

import re
from time import perf_counter
from uuid import uuid4

from rag.core.logger.cleaning_logger import CleaningLogger
from rag.core.schemas.ingestion_schemas import BlockType, ContentBlock, ParsedDocument
from rag.core.utils.text_processing import decode_and_normalize, normalize_text
from rag.cleaning.schemas import Cleaned_Document


_HARD_BLOCKS = {BlockType.FIGURE, BlockType.TABLE, BlockType.CODE, BlockType.FORMULA, BlockType.HEADING}
_TOC_RE = re.compile(r"\b(mục\s*lục|table\s+of\s+contents|contents)\b", re.IGNORECASE)
_PAGE_RE = re.compile(r"^(?:page|trang|p\.?)[\s.:#-]*\d+(?:\s*/\s*\d+)?$", re.IGNORECASE)
_FOOTER_RE = re.compile(r"^(?:footer|header|confidential|internal use only)\b", re.IGNORECASE)
_SENTENCE_END_RE = re.compile(r"[.!?:;。！？…]$|[)\]}\"']$")
_BULLET_RE = re.compile(r"^(?:[-*•]|\d+[.)])\s+")


class DocumentCleaner:
	"""Thực thi độc lập bốn process của cleaning."""

	def __init__(self, logger: CleaningLogger | None = None, trace_id: str | None = None) -> None:
		self.logger = logger or CleaningLogger()
		self.trace_id = trace_id or str(uuid4())

	def run(self, parsed: ParsedDocument) -> Cleaned_Document:
		started = perf_counter()
		process = "FILTERING_BLOCK"
		try:
			blocks, removed_count = self._filter_blocks(parsed.blocks)
			self._emit(
				process,
				"Đã lọc block không cần thiết cho embedding.",
				started,
				removed_block_count=removed_count,
				remaining_block_count=len(blocks),
			)

			
			process = "NORMALIZING"
			started = perf_counter()
			for block in blocks:
				if block.block_type not in _HARD_BLOCKS:
					block.raw_text = normalize_text(block.raw_text)
			self._emit(process, "Đã chuẩn hóa Unicode và text.", started, block_count=len(blocks))

			process = "RE_CLEANING"
			started = perf_counter()
			for block in blocks:
				if block.block_type not in _HARD_BLOCKS:
					block.raw_text = self._clean_text(block.raw_text)
					prefix = f"{block.hierarchical_path}\n" if block.hierarchical_path else ""
					block.contextual_text = f"{prefix}{block.raw_text}".strip()
			total_chars = sum(len(block.raw_text) for block in blocks)
			self._emit(process, "Đã làm sạch text bằng regex.", started, total_cleaned_chars=total_chars, block_count=len(blocks))

			metadata = parsed.metadata.model_copy(deep=True)
			metadata.length_count = len(blocks)
			metadata.char_count = total_chars
			return Cleaned_Document(
				metadata=metadata,
				blocks=blocks,
				total_cleaned_chars=total_chars,
				removed_block_count=removed_count,
			)
		except Exception as exc:
			self._emit_failure(process, started, exc)
			raise

	def _emit(self, process: str, message: str, started: float, **metrics: float) -> None:
		self.logger.emit(
			trace_id=self.trace_id,
			process=process,
			status="SUCCESS",
			message=message,
			metrics={"elapsed_ms": round((perf_counter() - started) * 1000, 2), **metrics},
		)

	def _emit_failure(self, process: str, started: float, exc: Exception) -> None:
		self.logger.emit(
			trace_id=self.trace_id,
			process=process,
			status="FAIL",
			message=f"Cleaning thất bại ở process {process}: {exc}",
			metrics={
				"elapsed_ms": round((perf_counter() - started) * 1000, 2),
				"error_type": type(exc).__name__,
			},
		)

	@staticmethod
	def _filter_blocks(blocks: list[ContentBlock]) -> tuple[list[ContentBlock], int]:
		kept: list[ContentBlock] = []
		removed = 0
		toc_level: int | None = None
		for block in blocks:
			if block.block_type == BlockType.HEADING and _TOC_RE.search(block.raw_text):
				toc_level = block.level
				removed += 1
				continue
			if toc_level is not None:
				if block.block_type == BlockType.HEADING and block.level is not None and block.level <= toc_level:
					toc_level = None
				else:
					removed += 1
					continue
			text = block.raw_text.strip()
			if _TOC_RE.search(block.hierarchical_path) or _PAGE_RE.fullmatch(text) or _FOOTER_RE.match(text):
				removed += 1
				continue
			kept.append(block)
		return kept, removed

	@staticmethod
	def _clean_text(text: str) -> str:
		text = text.replace("\r\n", "\n").replace("\r", "\n")
		text = re.sub(r"[ \t]+", " ", text)
		text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "-", text)
		lines = [line.strip() for line in text.split("\n") if line.strip()]
		output: list[str] = []
		for line in lines:
			if output and not _SENTENCE_END_RE.search(output[-1]) and not _BULLET_RE.match(line):
				output[-1] = f"{output[-1]} {line}"
			else:
				output.append(line)
		return re.sub(r"\n{3,}", "\n\n", "\n".join(output)).strip()

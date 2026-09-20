"""Split cleaned documents into heading-aware embedding chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from time import perf_counter
from uuid import uuid4

from rag.cleaning.schemas import Cleaned_Document
from rag.core.logger.chunking_logger import ChunkingLogger
from rag.core.schemas.chunking_schemas import Chunking_Block, Chunking_Document
from rag.core.schemas.ingestion_schemas import BlockType, ContentBlock


_HARD_BLOCKS = {BlockType.FIGURE, BlockType.TABLE, BlockType.CODE, BlockType.FORMULA}
_BULLET_RE = re.compile(r"(?m)^(?:\s*(?:[-*+]\s+|\d+[.)]\s+))")
_BOUNDARY_RE = re.compile(r"[.!?:;。！？…](?:\s|$)")


@dataclass(frozen=True)
class _Part:
    text: str
    hard: bool = False


class DocumentChunker:
    def __init__(
        self,
        *,
        max_chunk_size: int = 1000,
        overlap: int = 200,
        logger: ChunkingLogger | None = None,
        trace_id: str | None = None,
    ) -> None:
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be greater than zero")
        if overlap < 0 or overlap >= max_chunk_size:
            raise ValueError("overlap must be between zero and max_chunk_size - 1")
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.logger = logger or ChunkingLogger()
        self.trace_id = trace_id or str(uuid4())
        self._heading_splits = 0
        self._bullet_splits = 0
        self._recursive_splits = 0

    def run(self, cleaned: Cleaned_Document) -> Chunking_Document:
        started = perf_counter()
        try:
            self._heading_splits = self._bullet_splits = self._recursive_splits = 0
            chunks: list[Chunking_Block] = []
            self._split_range(cleaned.blocks, 0, len(cleaned.blocks), chunks)
            result = Chunking_Document(block=chunks, block_number=len(chunks))
            self._emit("SPLIT_HEADING", "Đã chia tài liệu theo heading.", started, split_count=self._heading_splits)
            self._emit("SPLIT_BULLET", "Đã chia và đóng gói theo bullet.", started, split_count=self._bullet_splits)
            self._emit("RECURSIVE_SPLIT", "Đã hoàn tất recursive fallback.", started, split_count=self._recursive_splits)
            return result
        except Exception as exc:
            self.logger.emit(
                trace_id=self.trace_id,
                process="RECURSIVE_SPLIT",
                status="FAIL",
                message=f"Chunking thất bại: {exc}",
                metrics={"elapsed_ms": round((perf_counter() - started) * 1000, 2)},
            )
            raise

    def _split_range(
        self,
        source_blocks: list[ContentBlock],
        start: int,
        end: int,
        chunks: list[Chunking_Block],
    ) -> None:
        if start >= end:
            return
        heading_level = self._lowest_heading_level(source_blocks, start, end)
        if heading_level is not None:
            groups = self._heading_groups(source_blocks, start, end, heading_level)
            if len(groups) > 1 or groups[0][0] > start:
                self._heading_splits += len(groups)
                for group_start, group_end in groups:
                    self._split_range(source_blocks, group_start, group_end, chunks)
                return

        content_blocks = [block for block in source_blocks[start:end] if block.block_type != BlockType.HEADING]
        if not content_blocks:
            return
        if self._size(content_blocks) <= self.max_chunk_size:
            chunks.append(self._make_chunk(content_blocks))
            return

        deeper_heading_level = self._lowest_heading_level(
            source_blocks,
            start,
            end,
            minimum_level=(heading_level + 1) if heading_level is not None else None,
        )
        if deeper_heading_level is not None:
            self._heading_splits += 1
            for group_start, group_end in self._heading_groups(source_blocks, start, end, deeper_heading_level):
                self._split_range(source_blocks, group_start, group_end, chunks)
            return

        if any(block.block_type == BlockType.LIST for block in content_blocks):
            self._split_bullets(content_blocks, chunks)
        else:
            self._split_recursive(content_blocks, chunks)

    @staticmethod
    def _lowest_heading_level(
        source_blocks: list[ContentBlock],
        start: int,
        end: int,
        minimum_level: int | None = None,
    ) -> int | None:
        levels = [
            block.level
            for block in source_blocks[start:end]
            if block.block_type == BlockType.HEADING and block.level is not None
            and (minimum_level is None or block.level >= minimum_level)
        ]
        return min(levels) if levels else None

    @staticmethod
    def _heading_groups(
        source_blocks: list[ContentBlock], start: int, end: int, heading_level: int
    ) -> list[tuple[int, int]]:
        boundaries = [
            index
            for index in range(start, end)
            if source_blocks[index].block_type == BlockType.HEADING
            and source_blocks[index].level == heading_level
        ]
        if not boundaries or boundaries[0] != start:
            boundaries.insert(0, start)
        return [(group_start, group_end) for group_start, group_end in zip(boundaries, boundaries[1:] + [end])]

    def _split_bullets(self, content_blocks: list[ContentBlock], chunks: list[Chunking_Block]) -> None:
        parts: list[_Part] = []
        for block in content_blocks:
            if block.block_type != BlockType.LIST:
                parts.append(_Part(block.raw_text, block.block_type in _HARD_BLOCKS))
                continue
            bullet_items = [item.strip() for item in _BULLET_RE.split(block.raw_text) if item.strip()]
            parts.extend(_Part(item) for item in (bullet_items or [block.raw_text]))

        self._bullet_splits += 1
        self._pack_parts(parts, content_blocks[0], chunks)

    def _pack_parts(self, parts: list[_Part], reference: ContentBlock, chunks: list[Chunking_Block]) -> None:
        pending: list[_Part] = []
        pending_size = 0
        for part in parts:
            part_size = len(part.text)
            if pending and pending_size + part_size + 2 > self.max_chunk_size:
                self._append_parts(pending, reference, chunks)
                pending = []
                pending_size = 0
            if part_size > self.max_chunk_size and not part.hard:
                self._append_recursive_text(part.text, reference, chunks)
            else:
                pending.append(part)
                pending_size += part_size + (2 if pending_size else 0)
        if pending:
            self._append_parts(pending, reference, chunks)

    def _split_recursive(self, content_blocks: list[ContentBlock], chunks: list[Chunking_Block]) -> None:
        self._recursive_splits += 1
        for block in content_blocks:
            if block.block_type in _HARD_BLOCKS:
                chunks.append(self._make_chunk([block]))
            else:
                self._append_recursive_text(block.raw_text, block, chunks)

    def _append_recursive_text(self, text: str, reference: ContentBlock, chunks: list[Chunking_Block]) -> None:
        text = text.strip()
        if not text:
            return
        self._recursive_splits += 1
        for piece in self._recursive_pieces(text):
            chunks.append(self._make_chunk([reference], raw_text=piece))

    def _recursive_pieces(self, text: str) -> list[str]:
        pieces: list[str] = []
        start = 0
        while start < len(text):
            if len(text) - start <= self.max_chunk_size:
                pieces.append(text[start:].strip())
                break
            limit = start + self.max_chunk_size
            boundary = max((match.end() for match in _BOUNDARY_RE.finditer(text, start, limit)), default=limit)
            if boundary <= start:
                boundary = limit
            pieces.append(text[start:boundary].strip())
            next_start = max(start + 1, boundary - self.overlap)
            start = next_start
        return [piece for piece in pieces if piece]

    def _append_parts(self, parts: list[_Part], reference: ContentBlock, chunks: list[Chunking_Block]) -> None:
        text = "\n\n".join(part.text for part in parts if part.text.strip()).strip()
        if text:
            chunks.append(self._make_chunk([reference], raw_text=text))

    def _make_chunk(self, source_blocks: list[ContentBlock], raw_text: str | None = None) -> Chunking_Block:
        reference = source_blocks[0]
        path = reference.hierarchical_path.strip()
        headings = [part.strip() for part in path.split(" > ") if part.strip()]
        text = raw_text if raw_text is not None else "\n\n".join(block.raw_text.strip() for block in source_blocks).strip()
        contextualized = f"{path}\n{text}".strip() if path else text
        return Chunking_Block(
            top_heading=headings[0] if headings else "",
            section=headings[1] if len(headings) > 1 else "",
            subsection=headings[2:],
            raw_text=text,
            contextualized_text=contextualized,
            hierachical_path=path,
        )

    @staticmethod
    def _size(blocks: list[ContentBlock]) -> int:
        return len("\n\n".join(block.raw_text.strip() for block in blocks))

    def _emit(self, process: str, message: str, started: float, **metrics: int) -> None:
        self.logger.emit(
            trace_id=self.trace_id,
            process=process,
            status="SUCCESS",
            message=message,
            metrics={"elapsed_ms": round((perf_counter() - started) * 1000, 2), **metrics},
        )
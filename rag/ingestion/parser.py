"""Bước DOCLING_PARSE: đọc tài liệu và dựng cây phân cấp ContentBlock."""

from __future__ import annotations

from io import BytesIO
from time import perf_counter
from typing import Any

from docling.document_converter import DocumentConverter
from docling_core.types.doc import (
    CodeItem,
    DoclingDocument,
    FormulaItem,
    ListItem,
    PictureItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
    TitleItem,
)

from rag.ingestion.base import BaseIngestion
from rag.ingestion.exceptions import ParseError
from rag.ingestion.schemas import BlockType, ContentBlock, DocumentMetadata, ParsedDocument
from rag.ingestion.validator import ValidatedSource


class DoclingParser(BaseIngestion):
    process = "DOCLING_PARSE"

    def __init__(self, *args: Any, converter: DocumentConverter | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.converter = converter or DocumentConverter()

    def run(self, source: ValidatedSource) -> ParsedDocument:
        started = perf_counter()
        try:
            conversion = self.converter.convert(str(source.local_path))
            docling_doc = conversion.document
            blocks = self._build_blocks(docling_doc)
            char_count = sum(len(block.raw_text) for block in blocks)
            page_num = self._page_count(docling_doc, blocks)

            parsed = ParsedDocument(
                metadata=DocumentMetadata(
                    source=source.origin,
                    page_num=page_num,
                    length_count=len(blocks),
                    char_count=char_count,
                    file_type=source.file_type,
                ),
                blocks=blocks,
            )
            self.emit(
                "SUCCESS",
                "Docling parse thành công.",
                self.timed_metrics(
                    started,
                    block_count=len(blocks),
                    page_num=page_num,
                    char_count=char_count,
                    figure_count=sum(1 for b in blocks if b.block_type == BlockType.FIGURE),
                    file_type=source.file_type,
                ),
            )
            return parsed
        except ParseError:
            raise
        except Exception as exc:
            self.fail(f"Docling không đọc được tài liệu: {exc}", self.timed_metrics(started), ParseError)

    def _build_blocks(self, doc: DoclingDocument) -> list[ContentBlock]:
        blocks: list[ContentBlock] = []
        heading_stack: list[tuple[int, str, str]] = []
        category_id = "uncategorized"

        for item, _doc_level in doc.iterate_items():
            block_type, raw_text, heading_level, image_data = self._extract_item(item, doc)
            if not raw_text and not image_data:
                continue

            if block_type == BlockType.HEADING and heading_level is not None:
                heading_level = self._normalize_heading_level(heading_level, heading_stack, item)
                while heading_stack and heading_stack[-1][0] >= heading_level:
                    heading_stack.pop()
                block = self._make_block(
                    raw_text=raw_text,
                    block_type=block_type,
                    page_num=self._item_page(item),
                    heading_stack=heading_stack,
                    category_id=category_id,
                    level=heading_level,
                )
                heading_stack.append((heading_level, raw_text.strip(), block.block_id))
                if heading_level == 1:
                    category_id = block.block_id
                    block.category_id = category_id
                blocks.append(block)
                continue

            blocks.append(
                self._make_block(
                    raw_text=raw_text,
                    block_type=block_type,
                    page_num=self._item_page(item),
                    heading_stack=heading_stack,
                    category_id=category_id,
                    level=heading_stack[-1][0] if heading_stack else None,
                    image_data=image_data,
                )
            )
        return blocks

    def _make_block(
        self,
        *,
        raw_text: str,
        block_type: BlockType,
        page_num: int | None,
        heading_stack: list[tuple[int, str, str]],
        category_id: str,
        level: int | None,
        image_data: bytes | None = None,
    ) -> ContentBlock:
        hierarchical_path = " > ".join(title for _, title, _ in heading_stack)
        section_id = heading_stack[-1][2] if heading_stack else "root"
        contextual_text = f"{hierarchical_path}\n{raw_text}".strip() if hierarchical_path else raw_text
        return ContentBlock(
            section_id=section_id,
            category_id=category_id,
            raw_text=raw_text,
            contextual_text=contextual_text,
            page_num=page_num,
            block_type=block_type,
            hierarchical_path=hierarchical_path,
            level=level,
            image_data=image_data,
        )

    def _extract_item(self, item: Any, doc: DoclingDocument) -> tuple[BlockType, str, int | None, bytes | None]:
        if isinstance(item, TitleItem):
            return BlockType.HEADING, (item.text or "").strip(), 1, None
        if isinstance(item, SectionHeaderItem):
            level = int(getattr(item, "level", 1) or 1)
            return BlockType.HEADING, (item.text or "").strip(), max(1, level), None
        if isinstance(item, PictureItem):
            caption = ""
            try:
                caption = (item.caption_text(doc) or "").strip()
            except Exception:
                caption = (getattr(item, "text", None) or "").strip()
            return BlockType.FIGURE, caption, None, self._picture_bytes(item, doc)
        if isinstance(item, TableItem):
            try:
                table_text = item.export_to_markdown(doc=doc)
            except TypeError:
                table_text = item.export_to_markdown()
            return BlockType.TABLE, (table_text or "").strip(), None, None
        if isinstance(item, ListItem):
            return BlockType.LIST, (item.text or "").strip(), None, None
        if isinstance(item, CodeItem):
            return BlockType.CODE, (item.text or "").strip(), None, None
        if isinstance(item, FormulaItem):
            return BlockType.FORMULA, (item.text or "").strip(), None, None
        if isinstance(item, TextItem):
            return BlockType.PARAGRAPH, (item.text or "").strip(), None, None
        text = (getattr(item, "text", None) or "").strip()
        return BlockType.OTHER, text, None, None

    @staticmethod
    def _normalize_heading_level(
        heading_level: int,
        heading_stack: list[tuple[int, str, str]],
        item: Any,
    ) -> int:
        """Title = 1; section/subsection lần lượt 2, 3..."""
        if isinstance(item, TitleItem):
            return 1
        if heading_stack and heading_stack[0][0] == 1 and heading_level == 1:
            return 2
        return heading_level

    def _picture_bytes(self, item: PictureItem, doc: DoclingDocument) -> bytes | None:
        try:
            image = item.get_image(doc)
        except Exception:
            return None
        if image is None:
            return None
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        return buffer.getvalue()

    @staticmethod
    def _item_page(item: Any) -> int | None:
        prov = getattr(item, "prov", None) or []
        if not prov:
            return None
        page_no = getattr(prov[0], "page_no", None)
        return int(page_no) if page_no is not None else None

    @staticmethod
    def _page_count(doc: DoclingDocument, blocks: list[ContentBlock]) -> int:
        pages = getattr(doc, "num_pages", None)
        if callable(pages):
            try:
                return int(pages())
            except Exception:
                pass
        if isinstance(pages, int) and pages > 0:
            return pages
        block_pages = [b.page_num for b in blocks if b.page_num]
        return max(block_pages) if block_pages else 0

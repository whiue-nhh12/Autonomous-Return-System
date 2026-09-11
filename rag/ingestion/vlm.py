"""Bước VLM_INFERENCE: trích xuất dữ liệu từ ảnh bằng OpenAI Vision."""

from __future__ import annotations

import base64
import os
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from rag.ingestion.base import BaseIngestion
from rag.ingestion.exceptions import VlmError
from rag.ingestion.schemas import BlockType, ContentBlock, ParsedDocument

_ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(_ENV_PATH)

MIN_IMAGE_WIDTH_PX = 150
MIN_IMAGE_HEIGHT_PX = 150


def _openai_api_key() -> str:
    return (os.getenv("OPENAI_API_KEY") or "").strip()


class VlmEnricher(BaseIngestion):
    process = "VLM_INFERENCE"

    def __init__(self, *args: Any, client: OpenAI | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._client = client

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            api_key = _openai_api_key()
            if not api_key:
                raise VlmError(
                    "Thiếu OPENAI_API_KEY trong .env.",
                    process=self.process,
                    metrics={"model": self.settings.vlm_model, "env_path": str(_ENV_PATH)},
                )
            self._client = OpenAI(api_key=api_key)
        return self._client

    def run(self, parsed: ParsedDocument) -> ParsedDocument:
        started = perf_counter()
        figure_blocks = [block for block in parsed.blocks if block.block_type == BlockType.FIGURE]
        if not figure_blocks:
            self.emit(
                "SUCCESS",
                "Không có ảnh cần VLM.",
                self.timed_metrics(started, image_count=0, model=self.settings.vlm_model),
            )
            return parsed

        eligible_blocks: list[ContentBlock] = []
        skipped = 0
        for block in figure_blocks:
            ok, width, height, reason = self._validate_image_for_vlm(block)
            if ok:
                eligible_blocks.append(block)
                continue
            skipped += 1
            self._apply_figure_text(block, block.raw_text or "Ảnh quá nhỏ, bỏ qua VLM.")
            self.emit(
                "WARNING",
                f"Bỏ qua VLM cho block {block.block_id}: {reason}",
                {
                    "block_id": block.block_id,
                    "page_num": block.page_num,
                    "width_px": width,
                    "height_px": height,
                    "min_width_px": MIN_IMAGE_WIDTH_PX,
                    "min_height_px": MIN_IMAGE_HEIGHT_PX,
                },
            )

        if not eligible_blocks:
            self.emit(
                "SUCCESS",
                "Không có ảnh đủ kích thước để gửi VLM.",
                self.timed_metrics(
                    started,
                    image_count=len(figure_blocks),
                    skipped_count=skipped,
                    sent_count=0,
                    model=self.settings.vlm_model,
                ),
            )
            parsed.metadata.length_count = len(parsed.blocks)
            parsed.metadata.char_count = sum(len(block.raw_text) for block in parsed.blocks)
            return parsed

        if not _openai_api_key() and self._client is None:
            self.fail(
                "Thiếu OPENAI_API_KEY trong .env.",
                self.timed_metrics(
                    started,
                    image_count=len(figure_blocks),
                    skipped_count=skipped,
                    sent_count=len(eligible_blocks),
                    model=self.settings.vlm_model,
                ),
                VlmError,
            )

        success = 0
        warnings = 0
        for block in eligible_blocks:
            try:
                extracted = self._infer_block(block)
                self._apply_figure_text(block, extracted)
                success += 1
            except Exception as exc:
                warnings += 1
                self._apply_figure_text(block, block.raw_text or "Không trích xuất được nội dung ảnh.")
                self.emit(
                    "WARNING",
                    f"VLM thất bại cho block {block.block_id}: {exc}",
                    {
                        "block_id": block.block_id,
                        "model": self.settings.vlm_model,
                        "page_num": block.page_num,
                    },
                )

        status = "WARNING" if warnings or skipped else "SUCCESS"
        self.emit(
            status,
            "Hoàn tất VLM cho các block hình ảnh.",
            self.timed_metrics(
                started,
                image_count=len(figure_blocks),
                sent_count=len(eligible_blocks),
                skipped_count=skipped,
                success_count=success,
                warning_count=warnings,
                min_width_px=MIN_IMAGE_WIDTH_PX,
                min_height_px=MIN_IMAGE_HEIGHT_PX,
                model=self.settings.vlm_model,
                max_tokens=self.settings.vlm_max_tokens,
                temperature=self.settings.vlm_temperature,
            ),
        )
        parsed.metadata.length_count = len(parsed.blocks)
        parsed.metadata.char_count = sum(len(block.raw_text) for block in parsed.blocks)
        return parsed

    def _validate_image_for_vlm(self, block: ContentBlock) -> tuple[bool, int | None, int | None, str]:
        if not block.image_data:
            return False, None, None, "Không có dữ liệu ảnh."
        try:
            with Image.open(BytesIO(block.image_data)) as image:
                width, height = image.size
        except Exception as exc:
            return False, None, None, f"Không đọc được kích thước ảnh: {exc}"
        if width < MIN_IMAGE_WIDTH_PX or height < MIN_IMAGE_HEIGHT_PX:
            return (
                False,
                width,
                height,
                f"Ảnh {width}x{height}px nhỏ hơn ngưỡng {MIN_IMAGE_WIDTH_PX}x{MIN_IMAGE_HEIGHT_PX}px.",
            )
        return True, width, height, ""

    def _apply_figure_text(self, block: ContentBlock, text: str) -> None:
        wrapped = self._wrap_figure(text)
        prefix = f"{block.hierarchical_path}\n" if block.hierarchical_path else ""
        block.raw_text = wrapped
        block.contextual_text = f"{prefix}{wrapped}".strip()
        block.image_data = None

    def _infer_block(self, block: ContentBlock) -> str:
        if not block.image_data:
            return block.raw_text or "Không có dữ liệu ảnh để gửi VLM."

        image_b64 = base64.b64encode(block.image_data).decode("utf-8")
        response = self.client.chat.completions.create(
            model=self.settings.vlm_model,
            temperature=self.settings.vlm_temperature,
            max_tokens=self.settings.vlm_max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.settings.vlm_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                }
            ],
        )
        text = (response.choices[0].message.content or "").strip()
        return text or block.raw_text or "Ảnh không chứa dữ liệu đọc được."

    @staticmethod
    def _wrap_figure(text: str) -> str:
        inner = (text or "").strip()
        if inner.startswith("<figure>") and inner.endswith("</figure>"):
            return inner
        return f"<figure>\n{inner}\n</figure>"

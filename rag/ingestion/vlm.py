"""Bước VLM_INFERENCE: trích xuất dữ liệu từ ảnh bằng OpenAI Vision."""

from __future__ import annotations

import base64
from time import perf_counter
from typing import Any

from openai import OpenAI

from rag.ingestion.base import BaseIngestion
from rag.ingestion.exceptions import VlmError
from rag.ingestion.schemas import BlockType, ContentBlock, ParsedDocument


class VlmEnricher(BaseIngestion):
    process = "VLM_INFERENCE"

    def __init__(self, *args: Any, client: OpenAI | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._client = client

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            if not self.settings.openai_api_key:
                raise VlmError(
                    "Thiếu OPENAI_API_KEY trong .env.",
                    process=self.process,
                    metrics={"model": self.settings.vlm_model},
                )
            self._client = OpenAI(api_key=self.settings.openai_api_key)
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

        if not self.settings.openai_api_key and self._client is None:
            self.fail(
                "Thiếu OPENAI_API_KEY trong .env.",
                self.timed_metrics(started, image_count=len(figure_blocks), model=self.settings.vlm_model),
                VlmError,
            )

        success = 0
        warnings = 0
        for block in figure_blocks:
            try:
                extracted = self._infer_block(block)
                wrapped = self._wrap_figure(extracted)
                block.raw_text = wrapped
                prefix = f"{block.hierarchical_path}\n" if block.hierarchical_path else ""
                block.contextual_text = f"{prefix}{wrapped}".strip()
                block.image_data = None
                success += 1
            except Exception as exc:
                warnings += 1
                fallback = self._wrap_figure(block.raw_text or "Không trích xuất được nội dung ảnh.")
                block.raw_text = fallback
                prefix = f"{block.hierarchical_path}\n" if block.hierarchical_path else ""
                block.contextual_text = f"{prefix}{fallback}".strip()
                block.image_data = None
                self.emit(
                    "WARNING",
                    f"VLM thất bại cho block {block.block_id}: {exc}",
                    {
                        "block_id": block.block_id,
                        "model": self.settings.vlm_model,
                        "page_num": block.page_num,
                    },
                )

        status = "WARNING" if warnings else "SUCCESS"
        self.emit(
            status,
            "Hoàn tất VLM cho các block hình ảnh.",
            self.timed_metrics(
                started,
                image_count=len(figure_blocks),
                success_count=success,
                warning_count=warnings,
                model=self.settings.vlm_model,
                max_tokens=self.settings.vlm_max_tokens,
                temperature=self.settings.vlm_temperature,
            ),
        )
        parsed.metadata.length_count = len(parsed.blocks)
        parsed.metadata.char_count = sum(len(block.raw_text) for block in parsed.blocks)
        return parsed

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

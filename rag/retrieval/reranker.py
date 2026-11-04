"""Local cross-encoder reranking implementation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from rag.core.schemas.search_schemas import SearchHit


class Reranker(Protocol):
    def score(self, query: str, hits: Sequence[SearchHit]) -> list[float]:
        ...


class CrossEncoderReranker:
    """Cross-encoder backed by Transformers, loaded lazily on first use."""

    DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        batch_size: int = 16,
        max_length: int = 512,
        model: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        if not model_name.strip() or batch_size <= 0 or max_length <= 0:
            raise ValueError("model_name, batch_size, and max_length must be valid")
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = model
        self._tokenizer = tokenizer
        self._device = None

    def _load(self) -> tuple[Any, Any, Any]:
        if self._model is None or self._tokenizer is None:
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
            except ImportError as exc:
                raise ImportError("CrossEncoderReranker requires torch and transformers") from exc
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if self._tokenizer is None:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self._model is None:
                self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._model.to(self._device)
            self._model.eval()
        if self._device is None:
            import torch

            self._device = (
                next(self._model.parameters()).device
                if hasattr(self._model, "parameters")
                else torch.device("cpu")
            )
        return self._model, self._tokenizer, self._device

    def score(self, query: str, hits: Sequence[SearchHit]) -> list[float]:
        if not hits:
            return []
        model, tokenizer, device = self._load()
        try:
            import torch
        except ImportError as exc:
            raise ImportError("CrossEncoderReranker requires torch") from exc

        scores: list[float] = []
        for start in range(0, len(hits), self.batch_size):
            batch = hits[start : start + self.batch_size]
            inputs = tokenizer(
                [query] * len(batch),
                [hit.text for hit in batch],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(device)
            with torch.inference_mode():
                logits = model(**inputs).logits
            if logits.ndim == 2 and logits.shape[1] > 1:
                logits = logits[:, -1]
            scores.extend(float(value) for value in logits.reshape(-1).detach().cpu().tolist())
        return scores
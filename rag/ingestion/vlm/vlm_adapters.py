"""Adapters for vision-language model providers."""

from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from openai import OpenAI


class VlmAdapter(ABC):
    """Contract implemented by every VLM provider adapter."""

    def __init__(self, model: str, api_key: str) -> None:
        if not model.strip():
            raise ValueError("model không được để trống.")
        if not api_key.strip():
            raise ValueError("api_key không được để trống.")
        self.model = model
        self.api_key = api_key

    @abstractmethod
    def infer(
        self,
        image_data: bytes,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Extract text from an image using the configured VLM."""


class OpenAICompatibleVlmAdapter(VlmAdapter):
    """Adapter for OpenAI and providers exposing the OpenAI chat API."""

    def __init__(self, model: str, api_key: str, base_url: str | None = None) -> None:
        super().__init__(model, api_key)
        self.client = OpenAI(api_key=api_key, base_url=base_url or None)

    def infer(
        self,
        image_data: bytes,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        image_b64 = base64.b64encode(image_data).decode("utf-8")
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                }
            ],
        )
        return (response.choices[0].message.content or "").strip()


AdapterFactory = Callable[..., VlmAdapter]


class VlmAdapterFactory:
    """Resolves an adapter from a model name while allowing custom providers."""

    _adapters: dict[str, AdapterFactory] = {}

    @classmethod
    def register(cls, name: str, adapter_factory: AdapterFactory) -> None:
        cls._adapters[name.strip().lower()] = adapter_factory

    @classmethod
    def create(
        cls,
        model: str,
        api_key: str,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> VlmAdapter:
        provider = model.split("/", 1)[0].lower()
        adapter_factory = cls._adapters.get(provider, OpenAICompatibleVlmAdapter)
        return adapter_factory(model=model, api_key=api_key, base_url=base_url, **kwargs)

"""Adapters for vision-language model providers."""

from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, ClassVar



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
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAI adapter yêu cầu package openai. "
                "Cài bằng: pip install openai"
            ) from exc
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


class GeminiVlmAdapter(VlmAdapter):
    """Adapter for Google's Gemini Generative AI API."""

    def __init__(self, model: str, api_key: str, base_url: str | None = None) -> None:
        super().__init__(model, api_key)
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "Gemini adapter yêu cầu package google-genai. "
                "Cài bằng: pip install google-genai"
            ) from exc

        self._genai = genai
        self.client = genai.Client(api_key=api_key)
        self.model = model.split("/", 1)[1] if "/" in model else model

    def infer(
        self,
        image_data: bytes,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                prompt,
                self._genai.types.Part.from_bytes(
                    data=image_data,
                    mime_type="image/png",
                ),
            ],
            config=self._genai.types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        )
        return (response.text or "").strip()


AdapterFactory = Callable[..., VlmAdapter]


class VlmAdapterFactory:
    """Resolves an adapter from a model name while allowing custom providers."""

    _adapters: ClassVar[dict[str, AdapterFactory]] = {}

    @classmethod
    def register(cls, name: str, adapter_factory: AdapterFactory) -> None:
        cls._adapters[name.strip().lower()] = adapter_factory

    @classmethod
    def create(
        cls,
        model: str,
        api_key: str,
        base_url: str | None = None,
        provider: str | None = None,
        **kwargs: Any,
    ) -> VlmAdapter:
        normalized_model = model.strip().lower()
        provider_name = (provider or normalized_model.split("/", 1)[0]).lower()
        if provider is None and provider_name not in cls._adapters and normalized_model.startswith("gemini-"):
            provider_name = "gemini"
        adapter_factory = cls._adapters.get(provider_name, OpenAICompatibleVlmAdapter)
        return adapter_factory(model=model, api_key=api_key, base_url=base_url, **kwargs)


VlmAdapterFactory.register("gemini", GeminiVlmAdapter)
VlmAdapterFactory.register("google", GeminiVlmAdapter)
VlmAdapterFactory.register("genai", GeminiVlmAdapter)

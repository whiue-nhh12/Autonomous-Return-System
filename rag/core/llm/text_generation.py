"""Shared text-generation contract and provider adapters."""

from __future__ import annotations

from typing import Protocol


class TextGenerator(Protocol):
    def generate(self, prompt: str, *, max_tokens: int, timeout_seconds: float) -> str:
        ...


class OpenAITextGenerator:
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        response_format: dict | None = None,
        max_retries: int = 2,
    ) -> None:
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url or None,
            max_retries=max_retries,
        )
        self.response_format = response_format

    def generate(self, prompt: str, *, max_tokens: int, timeout_seconds: float) -> str:
        request = dict(
            model=self.model,
            temperature=0,
            max_tokens=max_tokens,
            timeout=timeout_seconds,
            messages=[{"role": "user", "content": prompt}],
        )
        if self.response_format is not None:
            request["response_format"] = self.response_format
        response = self.client.chat.completions.create(**request)
        return (response.choices[0].message.content or "").strip()


class GroqTextGenerator:
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        timeout_seconds: float = 5.0,
        response_format: dict | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("GROQ_API_KEY is required to use GroqTextGenerator")

        from groq import Groq

        self.model = model
        self.client = Groq(
            api_key=api_key,
        )
        self.timeout_seconds = timeout_seconds
        self.response_format = response_format

    def generate(self, prompt: str, *, max_tokens: int, timeout_seconds: float) -> str:
        request = dict(
            model=self.model,
            temperature=0,
            max_tokens=max_tokens,
            timeout=timeout_seconds if timeout_seconds is not None else self.timeout_seconds,
            messages=[{"role": "user", "content": prompt}],
        )
        if self.response_format is not None:
            request["response_format"] = self.response_format
            request["reasoning_format"] = "hidden"
        response = self.client.chat.completions.create(**request)
        return (response.choices[0].message.content or "").strip()


class GeminiTextGenerator:
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        timeout_seconds: float = 5.0,
    ) -> None:
        from google import genai

        self._genai = genai
        options = {"timeout": int(timeout_seconds * 1000)}
        if base_url:
            options["base_url"] = base_url
        self.client = genai.Client(api_key=api_key, http_options=genai.types.HttpOptions(**options))
        self.model = model.split("/", 1)[-1]

    def generate(self, prompt: str, *, max_tokens: int, timeout_seconds: float) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=self._genai.types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=0,
            ),
        )
        return (response.text or "").strip()

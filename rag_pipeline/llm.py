#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Any, Dict, List

from llama_index.core.llms import (
    CompletionResponse,
    CompletionResponseGen,
    CustomLLM,
    LLMMetadata,
)
from openai import OpenAI


class OpenAIChatLLM(CustomLLM):
    """Адаптер LlamaIndex CustomLLM для OpenAI-совместимого Chat Completions API.

    Оборачивает клиента OpenAI, чтобы использовать его внутри LlamaIndex
    как обычную LLM: поддерживает complete и stream_complete.
    """
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_tokens: int = 800,
        system_prompt: str = "Ты движок RAG. Отвечай строго по предоставленному контенту.",
        enable_thinking: bool = False,
    ) -> None:
        super().__init__()
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model_name
        self._temperature = float(temperature)
        self._top_p = float(top_p)
        self._max_tokens = int(max_tokens)
        self._system_prompt = system_prompt
        self._enable_thinking = bool(enable_thinking)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=f"openai-compat::{self._model}",
            temperature=self._temperature,
            num_output=self._max_tokens,
        )

    def _make_messages(self, user_prompt: str) -> List[Dict[str, str]]:
        """Формирует список сообщений (system + user) для Chat API."""
        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Синхронное получение единого текста ответа для переданного промпта."""
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=self._make_messages(prompt),
            extra_body={"enable_thinking": self._enable_thinking},
            temperature=self._temperature,
            top_p=self._top_p,
            max_tokens=self._max_tokens,
        )
        text = (resp.choices[0].message.content or "").strip()
        return CompletionResponse(text=text)

    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """Потоковая генерация: возвращает нарастающий ответ частями."""
        try:
            stream = self._client.chat.completions.create(
                model=self._model,
                messages=self._make_messages(prompt),
                extra_body={"enable_thinking": self._enable_thinking},
                temperature=self._temperature,
                top_p=self._top_p,
                max_tokens=self._max_tokens,
                stream=True,
            )
        except Exception:
            yield self.complete(prompt)
            return

        buffer = []
        for event in stream:
            try:
                delta = event.choices[0].delta.content or ""
            except Exception:
                delta = ""
            if delta:
                buffer.append(delta)
                yield CompletionResponse(text="".join(buffer))
        if buffer:
            yield CompletionResponse(text="".join(buffer))

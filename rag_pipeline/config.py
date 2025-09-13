#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class EmbeddingConfig:
    """Параметры модели эмбеддингов.

    - model_name: имя модели HuggingFace для векторизации текста
    - embed_batch_size: размер батча при построении эмбеддингов
    """
    model_name: str = "BAAI/bge-small-en-v1.5"
    embed_batch_size: int = 32


@dataclass
class VectorStoreConfig:
    """Параметры векторного хранилища (Weaviate).

    - index_name: имя индекса/класса в Weaviate
    - use_embedded: использовать ли встроенный (embedded) Weaviate
    - weaviate_url: URL удалённого Weaviate (если используется)
    - weaviate_api_key: API-ключ для удалённого Weaviate (опционально)
    """
    index_name: str = "RAGIndex"
    use_embedded: bool = True
    weaviate_url: Optional[str] = None
    weaviate_api_key: Optional[str] = None


@dataclass
class LLMConfig:
    """Параметры языковой модели (OpenAI-совместимый API).

    - base_url: базовый URL сервиса LLM
    - api_key: ключ доступа
    - model_name: имя модели
    - temperature, top_p, max_tokens: параметры генерации
    - system_prompt: системный промпт для роли system
    - enable_thinking: передавать ли спец.параметр enable_thinking
    """
    base_url: str = "http://localhost:8080/v1"
    api_key: str = "test"
    model_name: str = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
    temperature: float = 0.3
    top_p: float = 0.9
    max_tokens: int = 800
    system_prompt: str = "Ты движок RAG. Отвечай строго по предоставленному контенту."
    enable_thinking: bool = False


@dataclass
class IndexingConfig:
    """Параметры индексирования корпуса документов.

    - chunk_size: размер текстового чанка
    - chunk_overlap: перекрытие соседних чанков
    - metadata_fields_for_auto_filters: описание метаданных для авто-фильтров
    """
    chunk_size: int = 512
    chunk_overlap: int = 64
    metadata_fields_for_auto_filters: List[Tuple[str, str, str]] = field(
        default_factory=lambda: [
            ("source", "str", "Имя исходного файла для фрагмента."),
            ("doc_title", "str", "Заголовок документа или извлечённый титул."),
        ]
    )


@dataclass
class RetrievalConfig:
    """Параметры извлечения и реранкинга.

    - similarity_top_k: сколько кандидатов доставать из индекса
    - hybrid_alpha: смешивание BM25 и векторного поиска (0-1)
    - reranker_model: модель реранкера
    - reranker_top_n: финальное число отранжированных пассages
    """
    similarity_top_k: int = 15
    hybrid_alpha: float = 0.5
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_top_n: int = 4

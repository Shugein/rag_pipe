#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Optional, Tuple

from llama_index.core import PromptTemplate, get_response_synthesizer
from llama_index.core.llms import CustomLLM
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.settings import Settings
from llama_index.core.vector_stores.types import MetadataInfo, VectorStoreInfo
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core import VectorStoreIndex

from .config import IndexingConfig, RetrievalConfig


class AdvancedRAG:
    """Расширенный движок RAG: извлечение, реранкинг и синтез ответа.

    - auto-retriever: автоматически подбирает фильтры по метаданным и ищет (hybrid)
    - reranker: переупорядочивает кандидатов по релевантности
    - response synthesizer: строит строгий ответ на основе контекста
    """
    def __init__(self, index: VectorStoreIndex, llm: CustomLLM, idx_cfg: IndexingConfig, ret_cfg: RetrievalConfig) -> None:
        self._index = index
        self._llm = llm
        self._idx_cfg = idx_cfg
        self._ret_cfg = ret_cfg

        Settings.llm = llm

        self._reranker = SentenceTransformerRerank(
            model=self._ret_cfg.reranker_model,
            top_n=self._ret_cfg.reranker_top_n,
        )

        self._vector_store_info = VectorStoreInfo(
            content_info="General knowledge chunks from your corpus.",
            metadata_info=[
                MetadataInfo(name=n, type=t, description=desc) for (n, t, desc) in self._idx_cfg.metadata_fields_for_auto_filters
            ],
        )

        self._qa_prompt = PromptTemplate(
            (
                "Ты помощник, отвечающий строго по контексту. "
                "Если ответа нет в контексте — скажи, что не знаешь.\n"
                "Контекст:\n"
                "---------------------\n"
                "{context_str}\n"
                "---------------------\n"
                "Вопрос: {query_str}\n"
                "Краткий ответ (ссылайся на названия файлов/заголовки, если уместно):"
            )
        )
        self._response_synth = get_response_synthesizer(text_qa_template=self._qa_prompt)

        self._naive_engine = self._index.as_query_engine()
        self._advanced_engine: Optional[RetrieverQueryEngine] = None

    def _make_auto_retriever(self) -> VectorIndexAutoRetriever:
        """Создаёт автотребователь (retriever) с гибридным поиском и top_k из конфигурации."""
        return VectorIndexAutoRetriever(
            self._index,
            llm=self._llm,
            vector_store_info=self._vector_store_info,
            similarity_top_k=self._ret_cfg.similarity_top_k,
            vector_store_query_mode="hybrid",
            alpha=self._ret_cfg.hybrid_alpha,
            verbose=False,
        )

    def _ensure_advanced_engine(self) -> RetrieverQueryEngine:
        """Лениво инициализирует сложный QueryEngine с реранкером и синтезом ответа."""
        if self._advanced_engine is None:
            retriever = self._make_auto_retriever()
            self._advanced_engine = RetrieverQueryEngine(
                retriever=retriever,
                response_synthesizer=self._response_synth,
                node_postprocessors=[self._reranker],
            )
        return self._advanced_engine

    def query_with_sources(self, question: str, mode: str = "advanced") -> Tuple[str, List[Dict[str, Any]]]:
        """Выполняет запрос и возвращает ответ и список источников.

        - mode="naive": простой query_engine без реранкинга
        - mode="advanced": автотребователь + реранкер + кастомный промпт
        """
        if mode == "naive":
            resp = self._naive_engine.query(question)
        else:
            adv = self._ensure_advanced_engine()
            resp = adv.query(question)

        text = str(resp)
        sources: List[Dict[str, Any]] = []
        try:
            for sn in getattr(resp, "source_nodes", []) or []:
                meta = sn.node.metadata or {}
                sources.append({
                    "score": getattr(sn, "score", None),
                    "source": meta.get("source"),
                    "doc_title": meta.get("doc_title"),
                    "snippet": (sn.node.get_content().strip()[:300] if sn.node else None),
                })
        except Exception:
            pass
        return text, sources

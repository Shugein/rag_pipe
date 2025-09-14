#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
from typing import Any, Dict, List, Optional

import weaviate
from weaviate.embedded import EmbeddedOptions

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.weaviate import WeaviateVectorStore

from rag_pipeline.config import (
    EmbeddingConfig,
    IndexingConfig,
    RetrievalConfig,
    VectorStoreConfig,
)
from rag_pipeline.engine import AdvancedRAG
from rag_pipeline.indexer import RAGIndexer
from rag_pipeline.llm import OpenAIChatLLM


app = FastAPI(title="Advanced RAG API (OpenAI-compatible)", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class IngestRequest(BaseModel):
    """Тело запроса для индексации директории документов."""
    data_dir: str
    index_name: str # имя коллекции/класса в Weaviate.
    weaviate_url: Optional[str] = None
    weaviate_api_key: Optional[str] = None
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chunk_size: int = 512
    chunk_overlap: int = 64
    drop_if_exists: bool = True


class IngestResponse(BaseModel):
    """Ответ на операцию индексации: имя индекса, число нод и длительность."""
    index_name: str
    nodes_indexed: int
    took_ms: int
    detail: str = "ok"


class QueryRequest(BaseModel):
    """Тело запроса для выполнения вопроса к индексу.

    Содержит параметры LLM и извлечения, включая режим (naive/advanced).
    """
    index_name: str
    query: str
    weaviate_url: Optional[str] = None
    weaviate_api_key: Optional[str] = None
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    openai_base_url: str = "http://localhost:8080/v1"
    openai_net_url: str = 'http://95.131.149.43:8080/v1'
    openai_api_key: str = "test"
    llm_model: str = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
    llm_temperature: float = 0.3
    llm_top_p: float = 0.9
    llm_max_tokens: int = 800
    mode: str = Field("advanced", pattern="^(naive|advanced)$")
    hybrid_alpha: float = 0.5
    top_k: int = 15
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_top_n: int = 4


class QueryResponse(BaseModel):
    """Ответ на запрос: текст ответа, источники и время выполнения."""
    answer: str
    sources: List[Dict[str, Any]]
    took_ms: int
    params: Dict[str, Any]


@app.get("/health")
def health() -> Dict[str, str]:
    """Простой health-check эндпоинт для мониторинга/оркестраторов."""
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    """Индексирует директорию документов в указанный индекс Weaviate.

    Загружает файлы, режет на чанки и сохраняет векторный индекс. Поддерживает
    локальный (embedded) и удалённый Weaviate.
    """
    t0 = time.time()
    vs_cfg = VectorStoreConfig(
        index_name=req.index_name,
        use_embedded=(req.weaviate_url is None),
        weaviate_url=req.weaviate_url,
        weaviate_api_key=req.weaviate_api_key,
    )
    emb_cfg = EmbeddingConfig(model_name=req.embedding_model)
    idx_cfg = IndexingConfig(chunk_size=req.chunk_size, chunk_overlap=req.chunk_overlap)

    try:
        indexer = RAGIndexer(vs_cfg, emb_cfg, idx_cfg)
        _, n = indexer.build_index(req.data_dir, drop_if_exists=req.drop_if_exists)
        took_ms = int((time.time() - t0) * 1000)
        return IngestResponse(index_name=req.index_name, nodes_indexed=n, took_ms=took_ms)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    """Выполняет вопрос к ранее построенному индексу и возвращает ответ с источниками."""
    t0 = time.time()
    try:
        vs_cfg = VectorStoreConfig(
            index_name=req.index_name,
            use_embedded=(req.weaviate_url is None),
            weaviate_url=req.weaviate_url,
            weaviate_api_key=req.weaviate_api_key,
        )
        emb_cfg = EmbeddingConfig(model_name=req.embedding_model)

        if vs_cfg.use_embedded:
            client = weaviate.Client(embedded_options=EmbeddedOptions())
        else:
            auth = None
            if vs_cfg.weaviate_api_key:
                try:
                    from weaviate.auth import AuthApiKey as _AuthApiKey  # type: ignore
                except Exception:
                    from weaviate import AuthApiKey as _AuthApiKey  # type: ignore
                auth = _AuthApiKey(vs_cfg.weaviate_api_key)
            client = weaviate.Client(url=vs_cfg.weaviate_url, auth_client_secret=auth)
        vector_store = WeaviateVectorStore(weaviate_client=client, index_name=vs_cfg.index_name)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        Settings.embed_model = HuggingFaceEmbedding(model_name=emb_cfg.model_name)
        index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

        llm = OpenAIChatLLM(
            base_url=req.openai_base_url,
            api_key=req.openai_api_key,
            model_name=req.llm_model,
            temperature=req.llm_temperature,
            top_p=req.llm_top_p,
            max_tokens=req.llm_max_tokens,
            system_prompt="Ты движок RAG. Отвечай строго по предоставленному контенту.",
            enable_thinking=False,
        )

        ret_cfg = RetrievalConfig(
            similarity_top_k=req.top_k,
            hybrid_alpha=req.hybrid_alpha,
            reranker_model=req.reranker_model,
            reranker_top_n=req.reranker_top_n,
        )
        idx_cfg = IndexingConfig()

        rag = AdvancedRAG(index=index, llm=llm, idx_cfg=idx_cfg, ret_cfg=ret_cfg)
        answer, sources = rag.query_with_sources(req.query, mode=req.mode)

        took_ms = int((time.time() - t0) * 1000)
        return QueryResponse(
            answer=answer,
            sources=sources,
            took_ms=took_ms,
            params={
                "mode": req.mode,
                "top_k": req.top_k,
                "hybrid_alpha": req.hybrid_alpha,
                "reranker_model": req.reranker_model,
                "reranker_top_n": req.reranker_top_n,
                "embedding_model": req.embedding_model,
                "llm_model": req.llm_model,
                "openai_base_url": req.openai_base_url,
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)

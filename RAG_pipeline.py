#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced RAG Pipeline — OpenAI-Compatible + FastAPI
===================================================

Эта версия добавляет HTTP-API поверх пайплайна:

  POST /ingest   — индексация папки документов в Weaviate
  POST /query    — запрос к индексу (naive/advanced)
  GET  /health   — проверка состояния

Пример запуска сервера:
  uvicorn advanced_rag_pipeline_openai_compat_api:app --host 0.0.0.0 --port 8000
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------- LlamaIndex components ----------
from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
    get_response_synthesizer,
    PromptTemplate,
)
from llama_index.core.llms import CustomLLM, CompletionResponse, CompletionResponseGen, LLMMetadata
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.settings import Settings
from llama_index.core.vector_stores.types import MetadataInfo, VectorStoreInfo
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank
import weaviate
from weaviate.embedded import EmbeddedOptions
from llama_index.vector_stores.weaviate import WeaviateVectorStore

# ---------- OpenAI-compatible client ----------
from openai import OpenAI

# ---------- FastAPI ----------
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ======================
# Configuration objects
# ======================

@dataclass
class EmbeddingConfig:
    model_name: str = "BAAI/bge-small-en-v1.5"
    embed_batch_size: int = 32

@dataclass
class VectorStoreConfig:
    index_name: str = "RAGIndex"
    use_embedded: bool = True
    weaviate_url: Optional[str] = None
    weaviate_api_key: Optional[str] = None

@dataclass
class LLMConfig:
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
    chunk_size: int = 512
    chunk_overlap: int = 64
    metadata_fields_for_auto_filters: List[Tuple[str, str, str]] = field(default_factory=lambda: [
        ("source", "str", "Имя исходного файла для фрагмента."),
        ("doc_title", "str", "Заголовок документа или извлечённый титул."),
    ])

@dataclass
class RetrievalConfig:
    similarity_top_k: int = 6
    hybrid_alpha: float = 0.5
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_top_n: int = 3

# ==============
# LLM Adapter
# ==============

class OpenAIChatLLM(CustomLLM):
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
        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
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

# =======================
# Vector store utilities
# =======================

def make_weaviate_client(cfg: VectorStoreConfig) -> weaviate.Client:
    if cfg.use_embedded:
        return weaviate.Client(embedded_options=EmbeddedOptions())
    if not cfg.weaviate_url:
        raise RuntimeError("Remote Weaviate requested, но url не указан.")
    if cfg.weaviate_api_key:
        return weaviate.Client(
            url=cfg.weaviate_url,
            auth_client_secret=weaviate.auth.AuthApiKey(cfg.weaviate_api_key),
        )
    return weaviate.Client(url=cfg.weaviate_url)

# ===============
# Index building
# ===============

class RAGIndexer:
    def __init__(self, vs_cfg: VectorStoreConfig, emb_cfg: EmbeddingConfig, idx_cfg: IndexingConfig) -> None:
        self.vs_cfg = vs_cfg
        self.emb_cfg = emb_cfg
        self.idx_cfg = idx_cfg

        Settings.embed_model = HuggingFaceEmbedding(model_name=self.emb_cfg.model_name)

        self._client = make_weaviate_client(self.vs_cfg)
        self._vector_store = WeaviateVectorStore(weaviate_client=self._client, index_name=self.vs_cfg.index_name)
        self._storage_context = StorageContext.from_defaults(vector_store=self._vector_store)

    def _load_documents(self, data_dir: str) -> List[Document]:
        from pathlib import Path
        from llama_index.core import SimpleDirectoryReader
        p = Path(data_dir)
        if not p.exists():
            raise FileNotFoundError(f"Data dir not found: {p}")

        reader = SimpleDirectoryReader(
            input_dir=str(p),
            recursive=True,
            required_exts=[".txt", ".md", ".pdf", ".html", ".htm"],
        )
        docs = reader.load_data()
        for d in docs:
            d.metadata = d.metadata or {}
            d.metadata.setdefault("source", d.metadata.get("file_name") or d.metadata.get("filename") or "unknown")
            if "doc_title" not in d.metadata:
                first_line = (d.text or "").strip().splitlines()[0] if d.text else ""
                d.metadata["doc_title"] = first_line[:120] if first_line else d.metadata["source"]
        return docs

    def _to_nodes(self, docs: List[Document]):
        splitter = SentenceSplitter(chunk_size=self.idx_cfg.chunk_size, chunk_overlap=self.idx_cfg.chunk_overlap)
        return splitter.get_nodes_from_documents(docs)

    def build_index(self, data_dir: str, drop_if_exists: bool = True) -> Tuple[VectorStoreIndex, int]:
        docs = self._load_documents(data_dir)
        nodes = self._to_nodes(docs)
        if drop_if_exists and self._client.schema.exists(self.vs_cfg.index_name):
            self._client.schema.delete_class(self.vs_cfg.index_name)
        index = VectorStoreIndex(nodes, storage_context=self._storage_context)
        return index, len(nodes)

# ====================
# Querying components
# ====================

class AdvancedRAG:
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
        if self._advanced_engine is None:
            retriever = self._make_auto_retriever()
            self._advanced_engine = RetrieverQueryEngine(
                retriever=retriever,
                response_synthesizer=self._response_synth,
                node_postprocessors=[self._reranker],
            )
        return self._advanced_engine

    def query_with_sources(self, question: str, mode: str = "advanced") -> Tuple[str, List[Dict[str, Any]]]:
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

# ==============
# FastAPI App
# ==============

app = FastAPI(title="Advanced RAG API (OpenAI-compatible)", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class IngestRequest(BaseModel):
    data_dir: str
    index_name: str
    weaviate_url: Optional[str] = None
    weaviate_api_key: Optional[str] = None
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chunk_size: int = 512
    chunk_overlap: int = 64
    drop_if_exists: bool = True

class IngestResponse(BaseModel):
    index_name: str
    nodes_indexed: int
    took_ms: int
    detail: str = "ok"

class QueryRequest(BaseModel):
    index_name: str
    query: str
    weaviate_url: Optional[str] = None
    weaviate_api_key: Optional[str] = None
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    openai_base_url: str = "http://localhost:8080/v1"
    openai_api_key: str = "test"
    llm_model: str = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
    llm_temperature: float = 0.3
    llm_top_p: float = 0.9
    llm_max_tokens: int = 800
    mode: str = Field("advanced", pattern="^(naive|advanced)$")
    hybrid_alpha: float = 0.5
    top_k: int = 6
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_top_n: int = 3

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    took_ms: int
    params: Dict[str, Any]

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
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
    t0 = time.time()
    try:
        vs_cfg = VectorStoreConfig(
            index_name=req.index_name,
            use_embedded=(req.weaviate_url is None),
            weaviate_url=req.weaviate_url,
            weaviate_api_key=req.weaviate_api_key,
        )
        emb_cfg = EmbeddingConfig(model_name=req.embedding_model)

        client = make_weaviate_client(vs_cfg)
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
    uvicorn.run("advanced_rag_pipeline_openai_compat_api:app", host="0.0.0.0", port=8000, reload=False)

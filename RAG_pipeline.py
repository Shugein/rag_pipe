
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced RAG Pipeline (single-file version)
==========================================

Implements an advanced RAG architecture inspired by the provided notebook:
- Pluggable LLM via LlamaIndex CustomLLM (supports local HF/Unsloth or OpenAI)
- HuggingFace BGE embeddings
- Weaviate vector store (embedded or remote)
- Sentence-level node parsing
- Hybrid retrieval with alpha blending
- Auto-retrieval (LLM-inferred metadata filters) via VectorIndexAutoRetriever
- Cross-encoder re-ranking (BAAI/bge-reranker-base)
- Custom QA prompt + response synthesizer
- Naive and Advanced query engines
- Simple CLI for ingestion and querying

Install (examples):
  pip install "llama-index>=0.11.0" llama-index-embeddings-huggingface \
              weaviate-client weaviate-embedded sentence-transformers \
              transformers torch

Optional (auto-retrieval w/ OpenAI):
  pip install llama-index-llms-openai openai

Usage examples:
  # 1) Ingest a folder of .txt/.md/.pdf into an embedded Weaviate index
  python advanced_rag_pipeline.py ingest --data-dir ./data --index-name ProjectDocs

  # 2) Naive RAG query
  python advanced_rag_pipeline.py query --index-name ProjectDocs --query "What does the design doc say about auth?" --mode naive

  # 3) Advanced RAG query (auto metadata filters + hybrid + rerank)
  #    If you set OPENAI_API_KEY, auto-retrieval will use GPT-4o-mini by default.
  python advanced_rag_pipeline.py query --index-name ProjectDocs --query "How do we shard the DB?" --mode advanced

Notes:
- If you prefer a local LLM: pass --llm-provider local and set --llm-endpoint/--llm-model/--llm-max-tokens,
  and implement the LocalLLM.generate() stub to call your HF/Unsloth model. For brevity this file ships a simple
  HF pipeline-based generator (if transformers is installed) or a placeholder you can extend.
- For remote Weaviate, provide --weaviate-url and (if needed) --weaviate-api-key.

Author: ChatGPT (GPT-5 Thinking)
"""
from __future__ import annotations

import os
import json
import argparse
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

# -----------------------------
# LlamaIndex core & components
# -----------------------------
from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
    get_response_synthesizer,
    PromptTemplate,
)
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import CustomLLM, CompletionResponse, CompletionResponseGen, LLMMetadata
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.settings import Settings
from llama_index.core.vector_stores.types import MetadataInfo, VectorStoreInfo

# Embeddings & Reranker
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.postprocessor import SentenceTransformerRerank

# Vector store: Weaviate
import weaviate
from weaviate.embedded import EmbeddedOptions
from llama_index.vector_stores.weaviate import WeaviateVectorStore

# (Optional) OpenAI for auto-retrieval LLM
try:
    from llama_index.llms.openai import OpenAI as LlamaIndexOpenAI
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

# (Optional) Transformers for a simple local text-generation pipeline
try:
    from transformers import pipeline as hf_pipeline
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False


# ======================
# Configuration objects
# ======================

@dataclass
class EmbeddingConfig:
    model_name: str = "BAAI/bge-small-en-v1.5"   # Good trade-off. Consider 'bge-base' for bigger.
    embed_batch_size: int = 32


@dataclass
class VectorStoreConfig:
    index_name: str = "RAGIndex"
    use_embedded: bool = True
    weaviate_url: Optional[str] = None
    weaviate_api_key: Optional[str] = None
    # You can add other Weaviate-specific tuning knobs here if needed.


@dataclass
class LLMConfig:
    provider: str = "openai"             # "openai" | "local"
    model_name: str = "gpt-4o-mini"      # For OpenAI; change as you like
    temperature: float = 0.2
    max_tokens: int = 512                # Max generation tokens
    # For "local" provider you may specify optional params below for the HF pipeline:
    local_task: str = "text-generation"
    local_device: Optional[str] = None   # e.g. "cuda:0" or "cpu" (Transformers auto-selects if None)
    local_kwargs: Dict[str, Any] = field(default_factory=lambda: {"do_sample": True, "top_p": 0.9})


@dataclass
class IndexingConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64
    metadata_fields_for_auto_filters: List[Tuple[str, str, str]] = field(default_factory=lambda: [
        # (name, type, description) - you should align these to real metadata
        ("source", "str", "Source file name for the chunk."),
        ("doc_title", "str", "Document title or inferred title."),
    ])


@dataclass
class RetrievalConfig:
    similarity_top_k: int = 6
    hybrid_alpha: float = 0.5            # 0.0 -> pure sparse; 1.0 -> pure dense
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_top_n: int = 3


# ==============
# LLM Adapters
# ==============

class LocalHFLLM(CustomLLM):
    """
    Minimal Local HF LLM wrapper for LlamaIndex.
    Uses Transformers pipeline('text-generation') if available.
    For production, replace with your preferred local stack (Unsloth, vLLM, Ollama, TGI, etc.).
    """
    def __init__(
        self,
        model_name: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
        task: str = "text-generation",
        device: Optional[str] = None,
        pipeline_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self._model_name = model_name
        self._temperature = float(temperature)
        self._max_tokens = int(max_tokens)
        self._task = task
        self._device = device
        self._pipeline_kwargs = pipeline_kwargs or {}

        if not _HAS_TRANSFORMERS:
            raise RuntimeError("transformers not installed. `pip install transformers` to use LocalHFLLM.")
        self._pipe = hf_pipeline(task=self._task, model=self._model_name, device=self._device)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            model_name=f"local::{self._model_name}",
            temperature=self._temperature,
            num_output=self._max_tokens,
        )

    def _gen_once(self, prompt: str) -> str:
        out = self._pipe(
            prompt,
            max_new_tokens=self._max_tokens,
            temperature=self._temperature,
            **self._pipeline_kwargs,
        )
        # Transformers pipeline returns list[dict] with 'generated_text'
        if isinstance(out, list) and out and isinstance(out[0], dict) and "generated_text" in out[0]:
            return str(out[0]["generated_text"])
        # Fallback
        return str(out)

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        text = self._gen_once(prompt)
        return CompletionResponse(text=text)

    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        # Simple non-streaming fallback: yield once
        text = self._gen_once(prompt)
        yield CompletionResponse(text=text)


def make_llm(cfg: LLMConfig) -> CustomLLM:
    """Factory for LLM that plugs into LlamaIndex Settings."""
    provider = (cfg.provider or "openai").lower()

    if provider == "openai":
        if not _HAS_OPENAI:
            raise RuntimeError("OpenAI provider requested, but llama-index-llms-openai is not installed.")
        # OPENAI_API_KEY must be in env
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set in the environment.")
        return LlamaIndexOpenAI(model=cfg.model_name, temperature=cfg.temperature, max_tokens=cfg.max_tokens)

    if provider == "local":
        return LocalHFLLM(
            model_name=cfg.model_name,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            task=cfg.local_task,
            device=cfg.local_device,
            pipeline_kwargs=cfg.local_kwargs,
        )

    raise ValueError(f"Unknown LLM provider: {cfg.provider}")


# =======================
# Vector store utilities
# =======================

def make_weaviate_client(vs_cfg: VectorStoreConfig) -> weaviate.Client:
    if vs_cfg.use_embedded:
        return weaviate.Client(embedded_options=EmbeddedOptions())
    # Remote
    if not vs_cfg.weaviate_url:
        raise RuntimeError("Remote Weaviate requested but --weaviate-url not provided.")
    if vs_cfg.weaviate_api_key:
        return weaviate.Client(url=vs_cfg.weaviate_url, auth_client_secret=weaviate.auth.AuthApiKey(vs_cfg.weaviate_api_key))
    return weaviate.Client(url=vs_cfg.weaviate_url)


# ===============
# Index building
# ===============

class RAGIndexer:
    """
    Loads documents, chunks them, and builds a VectorStoreIndex in Weaviate.
    """
    def __init__(
        self,
        vs_cfg: VectorStoreConfig,
        emb_cfg: EmbeddingConfig,
        idx_cfg: IndexingConfig,
    ) -> None:
        self.vs_cfg = vs_cfg
        self.emb_cfg = emb_cfg
        self.idx_cfg = idx_cfg

        # Plug embeddings + (later) LLM into LlamaIndex global Settings
        Settings.embed_model = HuggingFaceEmbedding(model_name=self.emb_cfg.model_name)

        self._client = make_weaviate_client(self.vs_cfg)
        self._vector_store = WeaviateVectorStore(weaviate_client=self._client, index_name=self.vs_cfg.index_name)
        self._storage_context = StorageContext.from_defaults(vector_store=self._vector_store)

    def _load_documents(self, data_dir: Path) -> List[Document]:
        from llama_index.core import SimpleDirectoryReader

        reader = SimpleDirectoryReader(
            input_dir=str(data_dir),
            recursive=True,
            required_exts=[".txt", ".md", ".pdf", ".html", ".htm"],
        )
        docs = reader.load_data()

        # Enrich some simple metadata (source filename + derived title)
        for d in docs:
            # Ensure consistent metadata keys for auto-retrieval later
            d.metadata = d.metadata or {}
            d.metadata.setdefault("source", d.metadata.get("file_name") or d.metadata.get("filename") or "unknown")
            # Try to set a doc_title; fallback to source
            if "doc_title" not in d.metadata:
                # small heuristic: first non-empty line as title
                first_line = (d.text or "").strip().splitlines()[0] if d.text else ""
                d.metadata["doc_title"] = first_line[:120] if first_line else d.metadata["source"]
        return docs

    def _to_nodes(self, docs: List[Document]) -> List:
        splitter = SentenceSplitter(chunk_size=self.idx_cfg.chunk_size, chunk_overlap=self.idx_cfg.chunk_overlap)
        nodes = splitter.get_nodes_from_documents(docs)
        return nodes

    def build_index(self, data_dir: Union[str, Path], drop_if_exists: bool = True) -> VectorStoreIndex:
        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"Data dir not found: {data_dir}")

        docs = self._load_documents(data_dir)
        nodes = self._to_nodes(docs)

        # Reset Weaviate class if needed
        if drop_if_exists and self._client.schema.exists(self.vs_cfg.index_name):
            self._client.schema.delete_class(self.vs_cfg.index_name)

        index = VectorStoreIndex(nodes, storage_context=self._storage_context)
        return index


# ====================
# Querying components
# ====================

class AdvancedRAG:
    """
    Assembles Naive and Advanced RAG query engines.
    Advanced = Auto-retrieval (inferred metadata filters) + Hybrid vector search + Cross-encoder rerank + Custom QA prompt
    """
    def __init__(
        self,
        index: VectorStoreIndex,
        llm: CustomLLM,
        idx_cfg: IndexingConfig,
        ret_cfg: RetrievalConfig,
    ) -> None:
        self._index = index
        self._llm = llm
        self._idx_cfg = idx_cfg
        self._ret_cfg = ret_cfg

        # Register LLM globally for LlamaIndex internal prompts
        Settings.llm = llm

        # Prepare reranker
        self._reranker = SentenceTransformerRerank(
            model=self._ret_cfg.reranker_model,
            top_n=self._ret_cfg.reranker_top_n,
        )

        # Build vector store info for auto-retrieval (LLM-inferred filters)
        self._vector_store_info = VectorStoreInfo(
            content_info="General knowledge chunks from your corpus.",
            metadata_info=[
                MetadataInfo(name=n, type=t, description=desc) for (n, t, desc) in self._idx_cfg.metadata_fields_for_auto_filters
            ],
        )

        # Default QA prompt; customize to your domain. Few-shot examples can be appended here if helpful.
        self._qa_prompt = PromptTemplate(dedent("""\
            You are a precise assistant. Use ONLY the provided context to answer.
            If the answer is not in the context, say you don't know.
            Context:
            ---------------------
            {context_str}
            ---------------------
            Question: {query_str}
            Answer (be concise and cite relevant filenames/titles if helpful):
        """).strip())

        self._response_synth = get_response_synthesizer(text_qa_template=self._qa_prompt)

        # Naive engine
        self._naive_engine = self._index.as_query_engine()

        # Advanced engine: build lazily upon first use (because it needs an LLM for auto-retrieval)
        self._advanced_engine: Optional[RetrieverQueryEngine] = None

    def _make_auto_retriever(self) -> VectorIndexAutoRetriever:
        # Auto-retriever LLM: we can reuse `self._llm` if it supports function calling well enough;
        # many prefer an OpenAI model for best results. Here we just reuse the configured LLM.
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

    # -------------
    # Public API
    # -------------
    def query(self, question: str, mode: str = "advanced") -> str:
        if mode == "naive":
            resp = self._naive_engine.query(question)
            return str(resp)

        adv = self._ensure_advanced_engine()
        resp = adv.query(question)
        return str(resp)


# ==============
# CLI Entrypoint
# ==============

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Advanced RAG pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Ingest
    pi = sub.add_parser("ingest", help="Ingest a folder of documents into Weaviate index")
    pi.add_argument("--data-dir", required=True, type=str, help="Path to folder with .txt/.md/.pdf/etc.")
    pi.add_argument("--index-name", required=True, type=str, help="Weaviate class/index name")
    pi.add_argument("--weaviate-url", type=str, default=None, help="Remote Weaviate URL (omit for embedded)")
    pi.add_argument("--weaviate-api-key", type=str, default=None, help="Remote Weaviate API key")
    pi.add_argument("--embedding-model", type=str, default="BAAI/bge-small-en-v1.5")
    pi.add_argument("--chunk-size", type=int, default=512)
    pi.add_argument("--chunk-overlap", type=int, default=64)
    pi.add_argument("--drop-if-exists", action="store_true", help="Drop class if already exists")

    # Query
    pq = sub.add_parser("query", help="Query an existing index")
    pq.add_argument("--index-name", required=True, type=str)
    pq.add_argument("--weaviate-url", type=str, default=None)
    pq.add_argument("--weaviate-api-key", type=str, default=None)
    pq.add_argument("--embedding-model", type=str, default="BAAI/bge-small-en-v1.5")

    pq.add_argument("--llm-provider", type=str, default="openai", choices=["openai", "local"])
    pq.add_argument("--llm-model", type=str, default="gpt-4o-mini")
    pq.add_argument("--llm-max-tokens", type=int, default=512)
    pq.add_argument("--llm-temperature", type=float, default=0.2)

    pq.add_argument("--query", required=True, type=str)
    pq.add_argument("--mode", type=str, default="advanced", choices=["naive", "advanced"])

    pq.add_argument("--hybrid-alpha", type=float, default=0.5)
    pq.add_argument("--top-k", type=int, default=6)
    pq.add_argument("--reranker-model", type=str, default="BAAI/bge-reranker-base")
    pq.add_argument("--reranker-top-n", type=int, default=3)

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Common configs
    vs_cfg = VectorStoreConfig(
        index_name=args.index_name,
        use_embedded=(args.weaviate_url is None),
        weaviate_url=args.weaviate_url,
        weaviate_api_key=args.weaviate_api_key,
    )
    emb_cfg = EmbeddingConfig(model_name=getattr(args, "embedding_model", "BAAI/bge-small-en-v1.5"))

    if args.cmd == "ingest":
        idx_cfg = IndexingConfig(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        indexer = RAGIndexer(vs_cfg, emb_cfg, idx_cfg)
        indexer.build_index(args.data_dir, drop_if_exists=args.drop_if_exists)
        print(f"Ingestion complete into Weaviate class: {vs_cfg.index_name}")
        return

    if args.cmd == "query":
        # Build a "light" index handle that points to existing Weaviate class
        client = make_weaviate_client(vs_cfg)
        vector_store = WeaviateVectorStore(weaviate_client=client, index_name=vs_cfg.index_name)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Plug in embeddings (needed for query vectorization)
        Settings.embed_model = HuggingFaceEmbedding(model_name=emb_cfg.model_name)

        # Rebuild a VectorStoreIndex handle (no re-ingestion)
        index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

        # LLM for query-time
        llm_cfg = LLMConfig(
            provider=args.llm_provider,
            model_name=args.llm_model,
            temperature=args.llm_temperature,
            max_tokens=args.llm_max_tokens,
        )
        llm = make_llm(llm_cfg)

        # Retrieval config
        ret_cfg = RetrievalConfig(
            similarity_top_k=args.top_k,
            hybrid_alpha=args.hybrid_alpha,
            reranker_model=args.reranker_model,
            reranker_top_n=args.reranker_top_n,
        )
        idx_cfg = IndexingConfig()  # default metadata fields: source, doc_title

        rag = AdvancedRAG(index=index, llm=llm, idx_cfg=idx_cfg, ret_cfg=ret_cfg)
        answer = rag.query(args.query, mode=args.mode)
        print(answer)
        return


if __name__ == "__main__":
    main()

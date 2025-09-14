#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
from typing import List, Tuple

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.weaviate import WeaviateVectorStore

from .config import EmbeddingConfig, IndexingConfig, VectorStoreConfig
from .vectorstore import make_weaviate_client

class RAGIndexer:
    """Индексатор корпуса документов в Weaviate через LlamaIndex.

    1) Загружает документы из директории
    2) Режет их на чанки (узлы)
    3) Строит векторный индекс во входном векторном хранилище
    """
    def __init__(self, vs_cfg: VectorStoreConfig, emb_cfg: EmbeddingConfig, idx_cfg: IndexingConfig) -> None:
        self.vs_cfg = vs_cfg
        self.emb_cfg = emb_cfg
        self.idx_cfg = idx_cfg

        Settings.embed_model = HuggingFaceEmbedding(model_name=self.emb_cfg.model_name)

        # Локальный embedded или удалённый (Docker/K8s) — согласно конфигу
        self._client = make_weaviate_client(self.vs_cfg)
        self._vector_store = WeaviateVectorStore(weaviate_client=self._client, index_name=self.vs_cfg.index_name)
        self._storage_context = StorageContext.from_defaults(vector_store=self._vector_store)

    def _load_documents(self, data_dir: str) -> List[Document]:
        """Читает документы из директории и нормализует базовые метаданные.

        - source: имя файла/источника
        - doc_title: первая строка или имя источника
        """
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
        """Разбивает документы на фрагменты (nodes) с заданным размером и overlap."""
        splitter = SentenceSplitter(chunk_size=self.idx_cfg.chunk_size, chunk_overlap=self.idx_cfg.chunk_overlap)
        return splitter.get_nodes_from_documents(docs)

    def build_index(self, data_dir: str, drop_if_exists: bool = True) -> Tuple[VectorStoreIndex, int]:
        """Строит/перестраивает индекс по директории данных.

        - drop_if_exists=True удаляет существующий класс в Weaviate
        - возвращает кортеж (индекс, количество нод)
        """
        docs = self._load_documents(data_dir)
        nodes = self._to_nodes(docs)
        if drop_if_exists and self._client.schema.exists(self.vs_cfg.index_name):
            self._client.schema.delete_class(self.vs_cfg.index_name)
        index = VectorStoreIndex(nodes, storage_context=self._storage_context)
        return index, len(nodes)

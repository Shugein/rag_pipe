"""
Тесты эндпоинта /query FastAPI-приложения.

Сценарии:
- Быстрый тест (моки Weaviate/Index/LLM/RAG) — проверяет проводку и формат ответа
- (Опционально) Интеграционный тест с embedded Weaviate — запускать по флагу

Запуск тестов:
  pytest -q tests/test_api_query.py

Интеграционный тест (медленный, требует зависимостей):
  RAG_RUN_INTEGRATION=1 pytest -q tests/test_api_query.py -k integration

Пример ручного запроса (после запуска uvicorn app.main:app):
  curl -X POST http://localhost:8000/query \
       -H 'Content-Type: application/json' \
       -d '{
             "index_name": "RAGIndex",
             "query": "Что такое RAG?",
             "mode": "advanced",
             "top_k": 6,
             "hybrid_alpha": 0.5
           }'
"""

import os
from pathlib import Path
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from app.main import app


class _DummyVectorStore:
    def __init__(self, weaviate_client: Any, index_name: str) -> None:
        self.client = weaviate_client
        self.index_name = index_name


class _DummyStorageContext:
    @classmethod
    def from_defaults(cls, vector_store: Any) -> "_DummyStorageContext":  # type: ignore[name-defined]
        inst = cls()
        inst.vector_store = vector_store
        return inst


class _DummyIndex:
    @staticmethod
    def from_vector_store(vector_store: Any, storage_context: Any) -> "_DummyIndex":  # type: ignore[name-defined]
        return _DummyIndex()


class _DummyLLM:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


class _DummyRAG:
    def __init__(self, index: Any, llm: Any, idx_cfg: Any, ret_cfg: Any) -> None:
        self.index = index
        self.llm = llm
        self.idx_cfg = idx_cfg
        self.ret_cfg = ret_cfg

    def query_with_sources(self, question: str, mode: str = "advanced"):
        answer = f"[dummy-answer] {question}"
        sources: List[Dict[str, Any]] = [
            {"score": 0.42, "source": "doc1.txt", "doc_title": "Dummy", "snippet": "Sample snippet"}
        ]
        return answer, sources


class _DummyEmbed:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass


def test_query_happy_path_mocked(monkeypatch) -> None:
    # Подменяем тяжёлые зависимости на простые моки
    from app import main as app_main
    monkeypatch.setattr(app_main, "make_weaviate_client", lambda cfg: object())
    monkeypatch.setattr(app_main, "WeaviateVectorStore", _DummyVectorStore)
    monkeypatch.setattr(app_main, "StorageContext", _DummyStorageContext)
    monkeypatch.setattr(app_main, "VectorStoreIndex", _DummyIndex)
    monkeypatch.setattr(app_main, "OpenAIChatLLM", _DummyLLM)
    monkeypatch.setattr(app_main, "AdvancedRAG", _DummyRAG)
    monkeypatch.setattr(app_main, "HuggingFaceEmbedding", _DummyEmbed)

    client = TestClient(app)
    payload = {
        "index_name": "TestIndex",
        "query": "Что такое RAG?",
        "mode": "advanced",
    }
    resp = client.post("/query", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "answer" in data and data["answer"].startswith("[dummy-answer]")
    assert isinstance(data.get("sources"), list) and len(data["sources"]) >= 1
    assert isinstance(data.get("took_ms"), int)
    assert isinstance(data.get("params"), dict)


@pytest.mark.integration
def test_query_integration_embedded_weaviate_naive(monkeypatch, tmp_path: Path) -> None:
    """Интеграционный тест (запускайте по флагу RAG_RUN_INTEGRATION=1).

    - Индексирует пару файлов (эндпоинт /ingest) — с подменой эмбеддера на мок
    - Выполняет запрос (эндпоинт /query, mode=naive) — с подменой LLM и реранкера
    """
    if os.environ.get("RAG_RUN_INTEGRATION") != "1":
        pytest.skip("Set RAG_RUN_INTEGRATION=1 to run this test")

    # Готовим корпус
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "doc1.txt").write_text("Столица Франции — Париж. RAG: retrieval augmented generation.")

    # Моки для лёгкого прогона: не скачиваем модели
    import rag_pipeline.indexer as idx_mod
    monkeypatch.setattr(idx_mod, "HuggingFaceEmbedding", _DummyEmbed)

    # Индексация
    client = TestClient(app)
    index_name = "TIndex"
    ingest_payload = {
        "data_dir": str(docs_dir),
        "index_name": index_name,
        "drop_if_exists": True,
    }
    resp_ingest = client.post("/ingest", json=ingest_payload)
    assert resp_ingest.status_code == 200, resp_ingest.text
    assert resp_ingest.json()["nodes_indexed"] > 0

    # Подменяем LLM и реранкер и эмбеддер на заглушки в /query
    from app import main as app_main
    monkeypatch.setattr(app_main, "OpenAIChatLLM", _DummyLLM)

    import rag_pipeline.engine as eng_mod
    class _NoopRerank:
        def __init__(self, model: str, top_n: int) -> None:
            self.top_n = top_n

        def postprocess_nodes(self, nodes, query_str=None):
            return nodes[: self.top_n]

    monkeypatch.setattr(eng_mod, "SentenceTransformerRerank", _NoopRerank)
    monkeypatch.setattr(app_main, "HuggingFaceEmbedding", _DummyEmbed)

    # Запрос (naive)
    query_payload = {
        "index_name": index_name,
        "query": "Какая столица Франции?",
        "mode": "naive",
    }
    resp_query = client.post("/query", json=query_payload)
    assert resp_query.status_code == 200, resp_query.text
    data = resp_query.json()
    assert isinstance(data.get("answer"), str)
    assert isinstance(data.get("sources"), list)

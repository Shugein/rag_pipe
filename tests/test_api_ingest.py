"""
Тесты эндпоинта /ingest FastAPI-приложения.

Сценарии:
- Успешная индексация (мок индексатора, без реального Weaviate/HF)
- Ошибка индексации (исключение из индексатора -> HTTP 500)

Запуск тестов:
  pytest -q tests/test_api_ingest.py

Ручная проверка эндпоинта (после запуска uvicorn app.main:app):
  curl -X POST http://localhost:8000/ingest \
       -H 'Content-Type: application/json' \
       -d '{
             "data_dir": "./data/docs",
             "index_name": "RAGIndex",
             "embedding_model": "BAAI/bge-small-en-v1.5",
             "chunk_size": 512,
             "chunk_overlap": 64,
             "drop_if_exists": true
           }'
"""

from pathlib import Path
from typing import Any, Tuple

import pytest
from fastapi.testclient import TestClient

from app.main import app


class _DummyIndexer:
    """Простой мок индексатора, без внешних зависимостей."""

    def __init__(self, vs_cfg: Any, emb_cfg: Any, idx_cfg: Any) -> None:
        self.vs_cfg = vs_cfg
        self.emb_cfg = emb_cfg
        self.idx_cfg = idx_cfg
        self._calls = []

    def build_index(self, data_dir: str, drop_if_exists: bool = True) -> Tuple[object, int]:
        self._calls.append((data_dir, drop_if_exists))
        # Возвращаем фиктивный индекс и число нод
        return object(), 7


def test_ingest_happy_path_mocked(monkeypatch, tmp_path: Path) -> None:
    # Подготавливаем временную директорию с документами
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / "doc1.txt").write_text("RAG test document one.")
    (docs_dir / "doc2.md").write_text("Another sample file.")

    # Подменяем класс RAGIndexer на мок, чтобы не тянуть HF/Weaviate
    from app import main as app_main

    monkeypatch.setattr(app_main, "RAGIndexer", _DummyIndexer)

    client = TestClient(app)
    payload = {
        "data_dir": str(docs_dir),
        "index_name": "TestIndex",
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "chunk_size": 256,
        "chunk_overlap": 32,
        "drop_if_exists": True,
    }
    resp = client.post("/ingest", json=payload)

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["index_name"] == payload["index_name"]
    assert isinstance(data["nodes_indexed"], int) and data["nodes_indexed"] > 0
    assert isinstance(data["took_ms"], int) and data["took_ms"] >= 0


def test_ingest_error_translates_to_500(monkeypatch, tmp_path: Path) -> None:
    # Мок, выбрасывающий исключение при индексации
    class _FailIndexer(_DummyIndexer):
        def build_index(self, data_dir: str, drop_if_exists: bool = True):  # type: ignore[override]
            raise FileNotFoundError("Data dir not found")

    from app import main as app_main
    monkeypatch.setattr(app_main, "RAGIndexer", _FailIndexer)

    client = TestClient(app)
    payload = {
        "data_dir": str(tmp_path / "missing"),
        "index_name": "TestIndex",
    }
    resp = client.post("/ingest", json=payload)

    assert resp.status_code == 500
    assert "detail" in resp.json()

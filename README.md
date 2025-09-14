# RAG Pipeline (modularized)

Простой и понятный рефакторинг исходного монолитного файла на модульную структуру.

## Структура

```
app/
  __init__.py
  main.py              # FastAPI-приложение и HTTP-эндпойнты

rag_pipeline/
  __init__.py
  config.py            # dataclass-конфиги (Embedding/VectorStore/Indexing/Retrieval/LLM)
  llm.py               # OpenAI-совместимый адаптер LLM (CustomLLM)
  vectorstore.py       # утилита создания клиента Weaviate
  indexer.py           # индексатор: чтение, чанкинг, построение индекса
  engine.py            # AdvancedRAG: retriever + reranker + synthesizer

RAG_pipeline.py        # Тонкий лаунчер: перенаправляет на app.main:app
```

## Запуск

1) Установите зависимости (llama-index, fastapi, uvicorn, weaviate-client, openai и т.д.)
2) Запустите сервер:

```
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Локальный Weaviate (Docker)

В составе `docker-compose.yml` есть сервис `weaviate`.

Запуск:
```
docker compose up -d weaviate
```

Подключение из API:
- В теле запросов `/ingest` и `/query` укажите `weaviate_url: "http://localhost:8080"`.
- Для локального embedded-режима опускайте поле `weaviate_url` (по умолчанию будет embedded).

Параметры сервиса:
- `DEFAULT_VECTORIZER_MODULE=none` — используем BYO-вектора (эмбеддинги из HuggingFace через LlamaIndex).
- Данные сохраняются в volume `weaviate_data`.

## Эндпойнты

- `GET /health` — проверка состояния
- `POST /ingest` — индексация директории в Weaviate
- `POST /query` — запрос к индексу (режимы: naive/advanced)

Параметры и поведение совпадают с исходной версией.

## Тестирование

- Быстрые тесты API (моки, без внешних зависимостей):
```
pytest -q tests/test_api_ingest.py
pytest -q tests/test_api_query.py -k happy
```

- Интеграционные (медленные) с embedded Weaviate и упрощёнными заглушками моделей:
```
RAG_RUN_INTEGRATION=1 pytest -q tests/test_api_query.py -k integration
```

Примеры обращений к API см. также в верхних docstring файлов `tests/test_api_ingest.py` и `tests/test_api_query.py`.

## Зависимости

Минимальный набор (пример):
```
pip install \
  fastapi uvicorn \
  llama-index-core llama-index-embeddings-huggingface \
  weaviate-client[embedded] \
  sentence-transformers \
  openai
```

Примечания:
- Для embedded Weaviate нужен `weaviate-client[embedded]`.
- При использовании Docker Weaviate достаточно `weaviate-client` без extras.

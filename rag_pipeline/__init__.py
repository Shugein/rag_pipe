"""Ядро RAG-пайплайна.

Содержит:
- config: dataclass-конфиги для эмбеддингов, векторного хранилища, индексирования и поиска
- llm: адаптер LlamaIndex CustomLLM для OpenAI‑совместимого Chat API
- vectorstore: фабрика клиента Weaviate (embedded/remote)
- indexer: загрузка документов, чанкинг и построение индекса
- engine: расширенный RAG-движок (retriever + reranker + synthesizer)
"""

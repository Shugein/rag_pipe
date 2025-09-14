#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Утилита создания клиента Weaviate (embedded или remote)."""

from __future__ import annotations

import weaviate
from weaviate.embedded import EmbeddedOptions

from .config import VectorStoreConfig


def make_weaviate_client(cfg: VectorStoreConfig) -> weaviate.Client:
    """Создаёт клиент Weaviate в зависимости от конфигурации.

    - embedded: локальный встроенный сервер Weaviate (без внешних сервисов)
    - remote: подключение к удалённому Weaviate (Docker/K8s) по URL, опционально с API‑ключом
    """
    if cfg.use_embedded:
        return weaviate.Client(embedded_options=EmbeddedOptions())
    if not cfg.weaviate_url:
        raise RuntimeError("Remote Weaviate запрошен, но URL не указан.")

    auth = None
    if cfg.weaviate_api_key:
        # Совместимость с разными версиями клиента
        try:
            from weaviate.auth import AuthApiKey as _AuthApiKey  # type: ignore
        except Exception:
            try:
                from weaviate import AuthApiKey as _AuthApiKey  # type: ignore
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("weaviate-client без поддержки AuthApiKey") from exc
        auth = _AuthApiKey(cfg.weaviate_api_key)

    return weaviate.Client(url=cfg.weaviate_url, auth_client_secret=auth)


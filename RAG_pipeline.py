#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Проект реорганизован: код разбит на модули пакета `rag_pipeline`,
а FastAPI-приложение вынесено в `app/main.py`.

Запуск сервера:
  uvicorn app.main:app --host 0.0.0.0 --port 8000
 
Этот файл служит тонким лаунчером для удобного запуска uvicorn.
"""

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)

from .config import config_services

import os
import uvicorn
from fastapi import FastAPI

from .api.routers import api_routers
from .logs import get_logger

logger = get_logger(__name__)

openapi_tags = [
    {
        "name": "Information",
        "description": "Справочная информация для подготовки конфигурации обучения: "
                       "пример конфига, доступные модели, оптимизаторы, планировщики, "
                       "метрики, аугментации, вычислительные устройства и health-check.",
    },
    {
        "name": "Validation",
        "description": "Проверка конфигурации обучения без запуска: соответствие схеме "
                       "и существование модели, оптимизатора, планировщика, метрик, "
                       "трансформаций, доступность устройства.",
    },
]

# Создание объекта fastapi
app = FastAPI(
    title="Trainer service",
    version="1.0.0",
    description="""
Сервис обучения DL моделей (классификация изображений).

Работает как воркер: опрашивает tasker, забирает задачи, обучает модель (PyTorch),
отправляет метрики в metrics и веса в ml_models. HTTP API предоставляет
справочную информацию для составления конфигурации обучения.

Воркер-цикл — в `app/core/worker.py`, запускается из `main.py` вместе с сервером.
""",
    openapi_tags=openapi_tags,
)

# Добавление роутеров
app.include_router(api_routers)

# Настройка конфига для запуска uvicorn
uv_conf = uvicorn.Config(
    app,
    host="0.0.0.0",
    port=int(os.getenv("TRAINER_SERVICE_PORT", 6200)),
)

# Создание объекта сервера uvicorn
server = uvicorn.Server(uv_conf)

logger.debug("Обьект сервера создан")

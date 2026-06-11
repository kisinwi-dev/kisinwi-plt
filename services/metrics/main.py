from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.health import check_health_all
from app.api.schemas import HealthStatus
from app.api.deps import cv_training_metric_manager, agents_metric_manager
from app.api.routers import routers
from app.logs import get_logger

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Проверка состояния требуемых БД
    # НЕ БЛОКИРУЕТ ЗАПУСК ЕСЛИ ТРЕБУЕМЫЕ БД НЕ РАБОТАЮТ
    health = check_health_all()

    cv_training_metric_manager.connect()
    agents_metric_manager.connect()

    if health.status == HealthStatus.HEALTHY:
        cv_training_metric_manager.ensure_indexes()
        agents_metric_manager.ensure_indexes()
    else:
        logger.warning("БД недоступна при старте — индексы не созданы")

    yield

    cv_training_metric_manager.disconnect()
    agents_metric_manager.disconnect()

# Обьект приложения
app = FastAPI(
    title="Metrics Service",
    version="0.1.0",
    description=(
        "Хранение и выдача метрик платформы KiSinWi.\n\n"
        "- **Метрики моделей** — значения по эпохам, разбитые по выборкам "
        "(train/val/test); пишутся trainer-сервисом раз в эпоху. "
        "Доступны сводная статистика, сравнение моделей и онлайн-поток (SSE).\n"
        "- **Метрики агентов** — использование LLM (токены и т.п.) по ответам, "
        "агрегируются по дискуссиям."
    ),
    openapi_tags=[
        {
            "name": "metrics",
            "description": "Метрики обучения моделей: запись по эпохам, выдача по выборкам, "
                           "SSE-поток, сводная статистика и сравнение моделей",
        },
        {
            "name": "agents",
            "description": "Метрики использования агентов: по ответам и сводно по дискуссиям",
        },
        {
            "name": "info",
            "description": "Служебные эндпоинты: состояние сервиса и зависимостей",
        },
    ],
    lifespan=lifespan,
)

app.include_router(routers)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=6310,
        reload=True,
    )

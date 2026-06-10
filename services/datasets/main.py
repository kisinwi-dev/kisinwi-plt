import asyncio
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from app.api.routers import api_routers
from app.core.filesystem import ArchiveManager
from app.logs import get_logger
from fastapi.middleware.cors import CORSMiddleware

logger = get_logger(__name__)

TEMP_TTL_HOURS = float(os.getenv("DATASETS_TEMP_TTL_HOURS", "24"))
TEMP_CLEANUP_INTERVAL_SECONDS = 3600

async def _temp_cleanup_loop():
    """Раз в час удаляет устаревшие загрузки из временной папки"""
    while True:
        try:
            ArchiveManager().cleanup_stale(TEMP_TTL_HOURS)
        except Exception:
            logger.exception("Ошибка фоновой очистки временной папки")
        await asyncio.sleep(TEMP_CLEANUP_INTERVAL_SECONDS)

@asynccontextmanager
async def lifespan(app: FastAPI):
    cleanup_task = asyncio.create_task(_temp_cleanup_loop())
    yield
    cleanup_task.cancel()

openapi_tags = [
    {
        "name": "Upload",
        "description": "Загрузка архивов с данными во временное хранилище. "
                       "Первый шаг перед созданием датасета или новой версии.",
    },
    {
        "name": "Datasets",
        "description": "CRUD датасетов: создание из загруженных данных, "
                       "получение метаданных, смена стандартной версии, удаление.",
    },
    {
        "name": "Versions",
        "description": "CRUD версий датасета: создание из загруженных данных, "
                       "получение метаданных, удаление.",
    },
    {
        "name": "Version Stats",
        "description": "Статистика версии по сплитам: количество изображений, "
                       "баланс и распределение классов, размеры и форматы изображений.",
    },
    {
        "name": "Version Compare",
        "description": "Сравнение двух версий датасета: дельты количества, "
                       "drift-метрики распределений (JS divergence, PSI), "
                       "баланс, размеры, по-файловый diff.",
    },
    {
        "name": "Info",
        "description": "Служебные эндпоинты: healthcheck сервиса и зависимостей.",
    },
]

app = FastAPI(
    title="Datasets Service",
    version="0.1.0",
    description="""
Сервис отвечает за управление датасетами и получение информации о них.
""",
    openapi_tags=openapi_tags,
    lifespan=lifespan,
)

app.include_router(
    api_routers
)

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
        port=6500,
        reload=True,
    )

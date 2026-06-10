from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.health import check_health_all
from app.api.deps import cv_training_metric_manager, agents_metric_manager
from app.api.routers import routers

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Проверка состояния требуемых БД
    # НЕ БЛОКИРУЕТ ЗАПУСК ЕСЛИ ТРЕБУЕМЫЕ БД НЕ РАБОТАЮТ
    check_health_all()

    cv_training_metric_manager.connect()
    agents_metric_manager.connect()

    yield

    cv_training_metric_manager.disconnect()
    agents_metric_manager.disconnect()

# Обьект приложения
app = FastAPI(
    title="Metrics Service",
    version="0.1.0",
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

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.health import check_health_all
from app.api.routers import routers
from app.api.exceptions import setup_exception_handlers


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Проверка состояния требуемых БД
    # НЕ БЛОКИРУЕТ ЗАПУСК ЕСЛИ ТРЕБУЕМЫЕ БД НЕ РАБОТАЮТ
    check_health_all()
    yield


app = FastAPI(
    title="Tasker Service",
    version="0.1.0",
    lifespan=lifespan
)

# добавление роутеров
app.include_router(routers)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Настройка выкидывания ошибок с бд и ошибки 500
setup_exception_handlers(app)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=6110,
        reload=True,
    )

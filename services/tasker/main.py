import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.health import check_health_all
from app.api.routers import routers
from app.api.exceptions import setup_exception_handlers

app = FastAPI(
    title="Tasker Service",
    version="0.1.0"
)

# добавление роутеров
app.include_router(routers)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":

    # Проверка состояния требуемых БД
    # НЕ БЛОКИРУЕТ ЗАПУСК ЕСЛИ ТРЕБУЕМЫЕ БД НЕ РАБОТАЮТ
    check_health_all()

    # Настройка выкидывания ошибок с бд и ошибки 500
    setup_exception_handlers(app)

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=6110,
        reload=True,
    )

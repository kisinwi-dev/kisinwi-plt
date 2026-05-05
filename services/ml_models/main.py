import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.core.health import check_health_all
from app.api.routers import routers
from app.logs import get_logger

logger = get_logger(__name__)

# Проверка состояния требуемых БД
# НЕ БЛОКИРУЕТ ЗАПУСК ЕСЛИ ТРЕБУЕМЫЕ БД НЕ РАБОТАЮТ
check_health_all()

app = FastAPI(
    title="ML Models Service",
    version="0.1.0"
)

logger.info("✅ app создано")

app.include_router(routers)

logger.info("✅ Добавлены эндпоинты")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Глобальная неизвестная ошибка: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Неизвестная внутренняя ошибка сервера."}
    )

logger.info("✅ Добавлены обработчики ошибок")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=6300,
        reload=True,
    )

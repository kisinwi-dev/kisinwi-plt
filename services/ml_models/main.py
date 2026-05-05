import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.health import check_health_all
from app.api.exceptions import setup_exception_handlers
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

setup_exception_handlers(app)
app.include_router(routers)

logger.info("✅ Добавлены эндпоинты")

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
        port=6300,
        reload=True,
    )

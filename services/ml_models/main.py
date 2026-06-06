import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.health import check_health_all
from app.api.exceptions import setup_exception_handlers
from app.api.routers import routers
from app.logs import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="ML Models Service",
    version="0.1.0"
)

app.include_router(routers)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Настройка обработчиков ошибок БД (503) и неизвестных ошибок (500).
# Должна выполняться на уровне модуля: при reload uvicorn импортирует "main:app"
# в воркере, где __name__ != "__main__".
setup_exception_handlers(app)

if __name__ == "__main__":
    # Проверка состояния требуемых БД
    # НЕ БЛОКИРУЕТ ЗАПУСК ЕСЛИ ТРЕБУЕМЫЕ БД НЕ РАБОТАЮТ
    check_health_all()

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=6300,
        reload=True,
    )

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from psycopg2 import OperationalError, InterfaceError, IntegrityError, DatabaseError

from app.logs import get_logger

logger = get_logger(__name__)

def setup_exception_handlers(app: FastAPI):
    
    logger.info("Настройка обработчиков ошибок...")

    @app.exception_handler(OperationalError)
    async def operational_error_handler(request: Request, exc: OperationalError):
        logger.error(f"Ошибка подключения к БД (OperationalError): {exc}", exc_info=True)
        return JSONResponse(
            status_code=503,
            content={"detail": "Ошибка подключения к базе данных."}
        )
    logger.info(" ✅ Обработчик 503 (OperationalError)")

    @app.exception_handler(InterfaceError)
    async def interface_error_handler(request: Request, exc: InterfaceError):
        logger.error(f"Ошибка подключения к БД (InterfaceError): {exc}", exc_info=True)
        return JSONResponse(
            status_code=503,
            content={"detail": "Ошибка подключения к базе данных."}
        )
    logger.info(" ✅ Обработчик 503 (InterfaceError)")

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Глобальная неизвестная ошибка: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Неизвестная внутренняя ошибка сервера."}
        )
    logger.info(" ✅ 500 ERROR")
    

    logger.info("✅ Добавлены обработчики ошибок")
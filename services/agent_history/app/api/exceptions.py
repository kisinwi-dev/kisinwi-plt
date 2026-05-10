from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.logs import get_logger

logger = get_logger(__name__)

def setup_exception_handlers(app: FastAPI):
    
    logger.info("Настройка обработчиков ошибок...")

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Глобальная неизвестная ошибка: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Неизвестная внутренняя ошибка сервера."}
        )
    logger.info(" ✅ Обработчик 500")
    

    logger.info("✅ Добавлены обработчики ошибок")
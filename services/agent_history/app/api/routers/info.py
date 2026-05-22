from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from app.api.schemas import SystemMessageType
from app.api.schemas import HealthResponse

routers = APIRouter(
    prefix="/info", 
    tags=["info"]
)

@routers.get(
    "/health",
    summary="Проверка работоспособности сервиса",
    description="Проверяет работу сервиса и зависимостей",
    response_model=HealthResponse,
    responses={
        200: {"description": "Выдаёт информацию о состояниях"},
        500: {"description": "Внутренняя ошибка сервера"},
    }
)
async def health():
    """Проверка подключения к БД"""
    try:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "healthy", "info": "No info"}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Неопределённая ошибка: {e}"
        )

@routers.get(
    "/type_system_mes",
    summary="Типы сообщений",
    description="Получить список возможных типов сообщений от сервиса агентов",
    responses={
        200: {"description": "Получен список типов"},
        500: {"description": "Внутренняя ошибка сервера"},
    }
)
async def get_type_mes_system():
    """Проверка подключения к БД"""
    try:
        types = [t.value for t in SystemMessageType]
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"types": types}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Неопределённая ошибка: {e}"
        )
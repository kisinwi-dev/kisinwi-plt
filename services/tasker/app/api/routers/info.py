from fastapi import APIRouter, HTTPException, status

from app.core.health import check_health_all
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
        info = check_health_all()
        return HealthResponse(
            status="healthy",
            info=info
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Неопределённая ошибка: {e}"
        )
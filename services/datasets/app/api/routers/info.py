from fastapi import APIRouter, HTTPException, status

from app.api.schemas import HealthResponse
from app.core.health import check_health_all

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
    try:
        return check_health_all()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Неопределённая ошибка: {e}"
        )
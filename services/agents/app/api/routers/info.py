from fastapi import APIRouter

from app.core.health import check_health_all
from app.api.schemas import HealthResponse

routers = APIRouter(
    tags=['info']
)

@routers.get(
    "/health",
    summary="Информация о сервисе",
    description="Вывод информации о работоспособности сервиса",
    response_model=HealthResponse
)
async def health():
    """Проверка работоспособности сервиса"""
    return check_health_all()


from fastapi import APIRouter

from app.core.health import check_health_all
from app.api.schemas import HealthResponse

routers = APIRouter(
    tags=['info']
)

@routers.get("/health")
async def health():
    """Проверка работоспособности сервиса"""
    return HealthResponse(
        status="healthy",
        info=check_health_all()
    )


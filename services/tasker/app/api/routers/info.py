from fastapi import APIRouter, HTTPException, status, Depends

from app.core.health import check_health_all
from app.api.schemas import HealthResponse, StatusesInfo, Status
from app.core.train_models_tasks import TrainingTaskManager
from app.api.deps import get_training_task_manager

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
        return check_health_all()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Неопределённая ошибка: {e}"
        )
    
@routers.get(
    "/statuses",
    response_model=StatusesInfo,
    summary="Получение списка статусов",
    description="Получение списка возможных значений статуса задач",
    responses={
        200: {"description": "Успешное получение списка статусов"},
        500: {"description": "Внутренняя ошибка сервера"},
        503: {"description": "Ошибка подключения к БД"}
    }
)
async def get_status(
    manager: TrainingTaskManager = Depends(get_training_task_manager)
):
    statuses = manager.get_status_values()
    return StatusesInfo(
        statuses=[
            Status(**status)
            for status in statuses
        ]
    )

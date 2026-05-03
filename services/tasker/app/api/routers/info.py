from fastapi import APIRouter, HTTPException, status, Depends

from app.core.health import check_health_all
from app.api.schemas import HealthResponse
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
    
@routers.get(
    "/statuses",
    summary="Получение списка статусов",
    description="Получение списка возможных значений статуса задач"
)
async def get_status(
    manager: TrainingTaskManager = Depends(get_training_task_manager)
):
    return manager.get_status_values()
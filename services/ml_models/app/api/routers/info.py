from psycopg2 import OperationalError, InterfaceError
from fastapi import APIRouter, HTTPException, Depends

from app.logs import get_logger
from app.core.health import check_health_all
from app.api.schemas import HealthResponse, MLModelsStatusesResponse, MLModelStatus
from app.api.deps import get_ml_models_manager, MlModelsManager

logger = get_logger(__name__)

routers = APIRouter(
    prefix="/info", 
    tags=["info"]
)

@routers.get(
    "/health",
    summary="Проверка работоспособности сервиса",
    description="""
Проверяет работу сервиса и зависимостей.
Выводит полную информацию о работоспособности сервиса
""",
    response_description="Информация о состоянии сервиса и его зависимостей",
    response_model=HealthResponse,
    responses={
        200: {"description": "Выдаёт информацию о состояниях"},
        500: {"description": "Внутренняя ошибка сервера"},
    }
)
async def health():
    """Проверка подключения к БД"""
    return check_health_all()

@routers.get(
    "/models/status",
    summary="Получение информации о статусах моделей",
    description="Возвращает справочник всех возможных статусов моделей с их описаниями",
    response_description="Список статусов моделей",
    response_model=MLModelsStatusesResponse,
    responses={
        200: {"description": "Выдаёт информацию о состояниях"},
        500: {"description": "Внутренняя ошибка сервера"},
        503: {"description": "Ошибка подключения к БД"}
    }
)
async def get_statuses(
    manager: MlModelsManager = Depends(get_ml_models_manager)
):
    try:
        statuses_list = manager.get_statuses_info()
        
        status_infos = [
            MLModelStatus(
                id=item["id"],
                status=item["status"],
                description=item["description"]
            )
            for item in statuses_list
        ]
        return MLModelsStatusesResponse(statuses=status_infos)
    
    except (OperationalError, InterfaceError) as e:
        raise HTTPException(
            status_code=503, 
            detail=f"Ошибка подключения к БД: {e}"
        )
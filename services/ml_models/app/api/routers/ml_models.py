from psycopg2 import OperationalError, InterfaceError, IntegrityError
from fastapi import APIRouter, Depends, HTTPException, status

from app.logs import get_logger
from app.api.schemas import *
from app.api.deps import get_ml_models_manager, MlModelsManager

routers = APIRouter(
    prefix='/models',
    tags=['models']
)

logger = get_logger(__name__)

@routers.post(
    "",
    summary="Создание модели",
    status_code=201,
    responses={
        201: {"description": "Модель успешно создана"},
        400: {"description": "Некорректные данные"},
        409: {"description": "Модель с таким name и version уже существует"},
        422: {"description": "Ошибка валидации данных"},
        500: {"description": "Внутренняя ошибка сервера"},
        503: {"description": "Ошибка подключения к БД"}
    }
)
async def create_model(
    model: MLModelCreate,
    manager: MlModelsManager = Depends(get_ml_models_manager)
):
    try:
        model_id = manager.create(
            **model.model_dump()
        )
        return {"model_id": model_id}
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Ошибка в запросе: {e}"
        )
    except (OperationalError, InterfaceError) as e:
        raise HTTPException(
            status_code=503, 
            detail=f"Ошибка подключения к БД: {e}"
        )
    except Exception as e:
        logger.error(f"Неожиданная ошибка при создании модели: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Внутрення ошибка сервиса"
        )


# @routers.delete(
#     "/{model_id}",
#     summary="Удаление моделей"
# )
# async def delete_task(
#     model_id: str,
#     manager: MlModelsManager = Depends(get_ml_models_manager)
# ):
#     return manager.delete(model_id)

# @routers.get(
#     "/{model_id}",
#     summary="Получить информацию о модели",
# )
# async def get_task_for_id(
#     model_id: str,
#     manager: MlModelsManager = Depends(get_ml_models_manager)
# ):
#     return manager.get_model(model_id)

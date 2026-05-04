from fastapi import APIRouter, Depends

from app.api.schemas import ModelCreate
from app.core.train_models_tasks import MlModelsManager
from app.api.deps import get_ml_models_manager

routers = APIRouter(
    prefix='/models',
    tags=['models']
)

@routers.post(
    "",
    summary="Создание модели"
)
async def create_task(
    model: ModelCreate, 
    manager: MlModelsManager = Depends(get_ml_models_manager)
):
    model_id = manager.create(
        name=model.name,
        model_type=model.model_type,
        description=model.description,
        classes=model.classes,
        train_params=model.train_params
    )
    return {"model_id": model_id}

@routers.post(
    "/count",
    summary="Количество моделей"
)
async def count_task(
    manager: MlModelsManager = Depends(get_ml_models_manager)
):
    return manager.count_models()


@routers.delete(
    "/{model_id}",
    summary="Удаление моделей"
)
async def delete_task(
    model_id: str,
    manager: MlModelsManager = Depends(get_ml_models_manager)
):
    return manager.delete(model_id)

@routers.get(
    "/{model_id}",
    summary="Получить информацию о модели",
)
async def get_task_for_id(
    model_id: str,
    manager: MlModelsManager = Depends(get_ml_models_manager)
):
    return manager.get_model(model_id)

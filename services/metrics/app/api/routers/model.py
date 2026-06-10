from typing import List

from fastapi import APIRouter, HTTPException, Depends

from app.api.schemas import ModelMetricAdd, ModelMetricAdds, ModelMetrics, ModelMetricsBatchRequest
from app.api.deps import get_cv_training_metrics_manager, CVMetricManager
from app.logs import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/models", tags=["metrics"])

@router.post(
    "/add",
    summary="Добавить метрику модели",
    description="Добавляет значения одной метрики модели в её выборку (train/val/test); "
                "выборка берётся из поля split или из префикса названия; "
                "при отсутствии модели создаёт запись",
    response_description="Статус операции",
)
async def add_metric(
    metric: ModelMetricAdd,
    manager: CVMetricManager = Depends(get_cv_training_metrics_manager)
):
    success = manager.add_metric(metric)
    if not success:
        raise HTTPException(status_code=500, detail="Ошибка добавления метрики")
    return {"status": "ok"}

@router.post(
    "/adds",
    summary="Добавить несколько метрик модели",
    description="Добавляет значения сразу нескольких метрик модели за один запрос (используется при обучении); "
                "каждая метрика попадает в свою выборку (train/val/test)",
    response_description="Статус операции",
)
async def add_metrics(
    metric: ModelMetricAdds,
    manager: CVMetricManager = Depends(get_cv_training_metrics_manager)
):
    success = manager.add_metrics(metric)
    if not success:
        raise HTTPException(status_code=500, detail="Ошибка добавления метрик")
    return {"status": "ok"}

@router.post(
    "/batch",
    response_model=List[ModelMetrics],
    summary="Получить метрики нескольких моделей",
    description="Возвращает метрики сразу нескольких моделей за один запрос, разбитые по выборкам "
                "(train/val/test); модели без метрик в ответ не попадают",
    response_description="Список метрик по каждой найденной модели",
)
async def get_models_metrics(
    body: ModelMetricsBatchRequest,
    manager: CVMetricManager = Depends(get_cv_training_metrics_manager)
):
    return manager.get_models_metrics(body.model_ids)

@router.get(
    "/{model_id}",
    response_model=ModelMetrics,
    summary="Получить метрики модели",
    description="Возвращает все метрики указанной модели с их значениями по эпохам, "
                "разбитые по выборкам (train/val/test)",
    response_description="Метрики модели",
)
async def get_task_metrics(
    model_id: str,
    manager: CVMetricManager = Depends(get_cv_training_metrics_manager)
):
    metrics = manager.get_model_metrics(model_id)
    if metrics is None:
        raise HTTPException(status_code=404, detail=f"Модель {model_id} не найдена")
    return metrics

@router.get(
    "/{model_id}/exists",
    summary="Проверить наличие метрик модели",
    description="Проверяет, есть ли в системе сохранённые метрики для указанной модели",
    response_description="Флаг наличия метрик модели",
)
async def task_exists(
    model_id: str,
    manager: CVMetricManager = Depends(get_cv_training_metrics_manager)
):
    exists = manager.model_metrics_exists(model_id)
    return {"model_id": model_id, "exists": exists}


@router.delete(
    "/{model_id}",
    summary="Удалить метрики модели",
    description="Удаляет все сохранённые метрики указанной модели из системы",
    response_description="Идентификатор модели и признак удаления",
)
async def task_delete(
    model_id: str,
    manager: CVMetricManager = Depends(get_cv_training_metrics_manager)
):
    deleted = manager.delete_metric(model_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Метрики модели {model_id} не найдены")
    return {"model_id": model_id, "deleted": True}
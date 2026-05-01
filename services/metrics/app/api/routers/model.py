from fastapi import APIRouter, HTTPException, Depends

from app.api.schemes import ModelMetricAdd, ModelMetricAdds, ModelMetrics
from app.api.deps import get_cv_training_metrics_manager, CVMetricManager
from app.logs import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/models", tags=["metrics"])

@router.post("/add")
async def add_metric(
    metric: ModelMetricAdd,
    manager: CVMetricManager = Depends(get_cv_training_metrics_manager)
):
    success = manager.add_metric(metric)
    if not success:
        raise HTTPException(status_code=500, detail="Ошибка добавления метрики")
    return {"status": "ok"}

@router.post("/adds")
async def add_metrics(
    metric: ModelMetricAdds,
    manager: CVMetricManager = Depends(get_cv_training_metrics_manager)
):
    success = manager.add_metrics(metric)
    if not success:
        raise HTTPException(status_code=500, detail="Ошибка добавления метрик")
    return {"status": "ok"}

@router.get("/{model_id}", response_model=ModelMetrics)
async def get_task_metrics(
    model_id: str,
    manager: CVMetricManager = Depends(get_cv_training_metrics_manager)
):
    metrics = manager.get_model_metrics(model_id)
    if metrics is None:
        raise HTTPException(status_code=404, detail=f"Модель {model_id} не найдена")
    return metrics

@router.get("/{model_id}/exists")
async def task_exists(
    model_id: str,
    manager: CVMetricManager = Depends(get_cv_training_metrics_manager)
):
    exists = manager.model_metrics_exists(model_id)
    return {"model_id": model_id, "exists": exists}


@router.delete("/{model_id}")
async def task_delete(
    model_id: str,
    manager: CVMetricManager = Depends(get_cv_training_metrics_manager)
):
    res = manager.model_metrics_exists(model_id)
    return {"is_delet": res}
from fastapi import APIRouter, HTTPException, Depends
from app.api.schemes import TrainingMetricAdd, TrainingMetricAdds, TaskTrainingMetrics
from app.api.deps import get_metrics_manager, CVMetricManager
from app.logs import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/metrics", tags=["metrics"])

@router.post("/add")
async def add_metric(
    metric: TrainingMetricAdd,
    manager: CVMetricManager = Depends(get_metrics_manager)
):
    success = manager.add_metric(metric)
    if not success:
        raise HTTPException(status_code=500, detail="Ошибка добавления метрики")
    return {"status": "ok"}

@router.post("/adds")
async def add_metrics(
    metric: TrainingMetricAdds,
    manager: CVMetricManager = Depends(get_metrics_manager)
):
    success = manager.add_metrics(metric)
    if not success:
        raise HTTPException(status_code=500, detail="Ошибка добавления метрик")
    return {"status": "ok"}

@router.get("/task/{task_id}", response_model=TaskTrainingMetrics)
async def get_task_metrics(
    task_id: str,
    manager: CVMetricManager = Depends(get_metrics_manager)
):
    metrics = manager.get_task_metrics(task_id)
    if metrics is None:
        raise HTTPException(status_code=404, detail=f"Задача {task_id} не найдена")
    return metrics

@router.get("/task/{task_id}/exists")
async def task_exists(
    task_id: str,
    manager: CVMetricManager = Depends(get_metrics_manager)
):
    exists = manager.task_exists(task_id)
    return {"task_id": task_id, "exists": exists}


@router.delete("/task/{task_id}")
async def task_delete(
    task_id: str,
    manager: CVMetricManager = Depends(get_metrics_manager)
):
    res = manager.delete_task(task_id)
    return {"is_delet": res}
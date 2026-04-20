import json
from fastapi import APIRouter, Query, HTTPException

from app.services.tasker import tasker
from app.core.ml.crews import run_search_params_json
from app.core.analytic.crews import run_analysis

routers = APIRouter()

@routers.get("/run_")
def health_status(
    dataset_id: str = Query(..., description="ID датасета для анализа"),
    version_id: str = Query(None, description="ID версии датасета"),
):
    """
    Анализ датасета и отправка задачи в сервис обучения.
    """
    try:
        # Анализ датасета
        result_analysis = run_analysis(dataset_id, version_id)
        
        # Подготовка JSON для задачи
        result_json = run_search_params_json(
            role="CV-enginer",
            previous_output=result_analysis.raw
        )

        # Отправка в сервис задач
        post_result = tasker.post_task(result_json.raw)
        
        # Возвращаем результат
        return {
            "status": "completed",
            "dataset_id": dataset_id,
            "version_id": version_id,
            "analysis_completed": True,
            "json_prepared": True,
            "task_submission": post_result,
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при выполнении: {str(e)}"
        )

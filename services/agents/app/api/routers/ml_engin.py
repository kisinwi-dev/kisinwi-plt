from fastapi import APIRouter, Query, HTTPException

from app.core.crews.ml_engine import run_ml_engineering
from app.core.crews.task_preparer import run_create_task_params_json

routers = APIRouter(
    tags=['engineering']
)

@routers.get(
        "/engine_reasoning",
        description="Рассуждения агентов ML-инженеров"
)
def run_etp(
    number_engineer: int = Query(1, description="Количество инженеров"),
    analysis_result: str = Query("", description="Дополнительная информация по имеющимся данным и задаче")
):
    try:  
        result, metrics = run_ml_engineering(
            number_engineer, 
            analysis_result
        )

        return {
            "number_engineer": number_engineer,
            "analysis": result,
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при выполнении: {str(e)}"
        )

@routers.get(
        "/create_task",
        description="Получение json для задачи тренировки. Json формирует агент."
)
def run_ctpj(
    dataset_id: str = Query(..., description="Датасет используемый при обучении"),
    version_id: str = Query(..., description="Версия датасета используемая при обучении"),
    previous_outputs: list = Query(..., description="Описание действий для достижения требуемого результата"),
):
    try:
        from app.services.tasker import tasker

        result, metrics = run_create_task_params_json(
            previous_outputs,
            dataset_id=dataset_id,
            version_id=version_id
        )
        result = tasker.post_task(result)

        return {
            "result": result,
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при выполнении: {str(e)}"
        )

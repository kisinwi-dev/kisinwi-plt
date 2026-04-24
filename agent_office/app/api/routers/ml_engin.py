from fastapi import APIRouter, Query

from app.core.crews.ml_engine import run_ml_engineering
from app.core.crews.task_preparer import run_create_task_params_json

routers = APIRouter(
    tags=['engineering']
)

@routers.get(
        "/engine_reasoning",
        description="Рассуждения ML инженеров"
)
def run_etp(
    number_engineer: int = Query(1, description="Количество инженеров"),
    analysis_result: str = Query("", description="Дополнительная информация по имеющимся данным и задаче")
):
    """Агенты ML-инженеры рассуждают над лучшей моделью"""
    
    result, metrics = run_ml_engineering(
        number_engineer, 
        analysis_result
    )
    
    return {
        "number_engineer": number_engineer,
        "analysis": result,
        "metrics": metrics
    }

@routers.get(
        "/create_task",
        description="Создание json для таск сервиса"
)
def run_ctpj(
    previous_outputs: list = Query(..., description="Итоги размышлений инженеров"),
):
    """Агент составляет результирующий json для отправки в сервис задач"""
    
    result, metrics = run_create_task_params_json(previous_outputs)
    
    return {
        "result": result,
        "metrics": metrics
    }


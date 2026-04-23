from fastapi import APIRouter, Query

from app.core.ml_engin.crews import run_engine_training_pipeline
from app.core.summarizer.crews import run_create_task_params_json

routers = APIRouter(
    tags=['engineering']
)

@routers.get(
        "/engine_reasoning",
        description="Рассуждения ML инженеров"
)
def run_etp(
    number_engineer: int = Query(1, description="Количество инженеров"),
    previous_output: str = Query("", description="Дополнительная информация по имеющимся данным и задаче")
):
    """
    Анализ датасета.
    
    Параметры:
    - dataset_id: ID датасета
    - version_id: ID версии
    """
    
    result = run_engine_training_pipeline(
        number_engineer, 
        previous_output
    )
    
    return {
        "number_engineer": number_engineer,
        "analysis": result.raw
    }

@routers.get(
        "/create_task",
        description="Создание json для таск сервиса"
)
def run_ctpj(
    previous_outputs: list = Query(..., description="Итоги размышлений инженеров"),
):
    """
    Анализ датасета.
    
    Параметры:
    - dataset_id: ID датасета
    - version_id: ID версии
    """
    
    result = run_create_task_params_json(previous_outputs)
    
    return {
        "result": result.raw
    }


from fastapi import APIRouter, Query
from app.core.ml_engin.crews import run_engine_training_pipeline
from app.core.summarizer.crews import run_create_task_params_json
routers = APIRouter()

@routers.get("/run_engine_training_pipeline")
def run_etp(
    role: str = Query(..., description="Роль агента (Пример:'CV инженер')"),
    number_engineer: int = Query(1, description="Номер инженера (инженеров может быть больше 1-ного)"),
    previous_output: str = Query("", description="Дополнительная информация по имеющимся данным и задаче")
):
    """
    Анализ датасета.
    
    Параметры:
    - dataset_id: ID датасета
    - version_id: ID версии
    """
    
    result = run_engine_training_pipeline(role, number_engineer, previous_output)
    
    return {
        "role": role,
        "number_engineer": number_engineer,
        "analysis": result.raw
    }

@routers.get("/run_create_task_params_json")
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


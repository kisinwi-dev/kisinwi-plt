from fastapi import APIRouter, Query
from app.core.ml.crews import run_search_params_json

routers = APIRouter()

@routers.get("/ml")
def health_status(
    dataset_info: str = Query(..., description="Информация о датасете"),
):
    """
    Анализ датасета.
    """
    
    result = run_search_params_json(
        role="CV-enginer",
        previous_output=dataset_info
    )
    
    return {
        "json": result.raw
    }

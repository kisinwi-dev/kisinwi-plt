from fastapi import APIRouter, Query
from app.service.crewai.analytic.crews import run_analysis

routers = APIRouter()

@routers.get("/analytic")
def health_status(
    dataset_id: str = Query(..., description="ID датасета для анализа"),
    version_id: str = Query(None, description="ID версии датасета")
):
    """
    Анализ датасета.
    
    Параметры:
    - dataset_id: ID датасета
    - version_id: ID версии
    """
    
    result = run_analysis(dataset_id, version_id)
    
    return {
        "dataset_id": dataset_id,
        "version_id": version_id,
        "analysis": result.raw
    }

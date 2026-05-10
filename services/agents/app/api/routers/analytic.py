from fastapi import APIRouter, Query, HTTPException

from app.core.crews.metrics_analyst import run_metrics_analyst
from app.core.crews.dataset_analyst import run_dataset_analyst

routers = APIRouter(
    prefix='/analytics',
    tags=['analytics']
)

@routers.get(
        "/datasets",
        description="Анализ датасета"
)
def analyze_metrics(
    discussion_id: str = Query(..., description="ID диалога"),
    dataset_id: str = Query(..., description="ID датасета"),
    dataset_version_id: str = Query(..., description="ID версии датасета"),
):
    try:
        result = run_dataset_analyst(
            discussion_id=discussion_id,
            dataset_id=dataset_id,
            dataset_version_id=dataset_version_id,
            verbose=True
        )
        
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при выполнении: {str(e)}"
        )


# @routers.get(
#         "/data",
#         description="Анализ датасета"
# )
# def analyze_dataset(
#     dataset_id: str = Query(..., description="ID датасета"),
#     version_id: str | None = Query(None, description="ID версии")
# ):
#     try:
#         result, metrics = data_analyse(dataset_id, version_id)
        
#         return {
#             "dataset_id": dataset_id,
#             "version_id": version_id,
#             "analysis": result,
#             "metrics": metrics
#         }
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Ошибка при выполнении: {str(e)}"
#         )

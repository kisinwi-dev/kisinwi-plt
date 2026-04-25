from fastapi import APIRouter, Query, HTTPException

from app.core import full_pipeline

routers = APIRouter(
    tags=['pipeline']
)

@routers.get(
        "/full_pipeline",
        description="Одна полная итерация. Анализ, рассуждение и запуск тренировки"
)
def fp(
    dataset_id: str = Query(..., description="ID датасета для анализа"),
    version_id: str = Query(None, description="ID версии датасета"),
    count_engine: int = Query("", description="Количество используемых агентов ML-инженеров")
):
    try:
        out = full_pipeline(
            dataset_id,
            version_id,
            count_engine
        )

        # Возвращаем результат
        return {
            "status": "completed", 
            **out
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при выполнении: {str(e)}"
        )

from fastapi import APIRouter, Query, HTTPException

from app.core import full_pipeline_agent

routers = APIRouter(
    tags=['pipeline']
)

@routers.get(
        "/full_pipeline",
        description="Одна полная итерация анализ, рассуждение и запуск тренировки."
)
def full_pipeline(
    dataset_id: str = Query(..., description="ID датасета для анализа"),
    version_id: str = Query(None, description="ID версии датасета"),
    count_engine: int = Query("", description="Количество используемых агентов-инженеров")
):
    """
    Анализ датасета и отправка задачи в сервис обучения.
    """
    try:
        full_pipeline_agent(
            dataset_id,
            version_id,
            task,
            count_engine
        )
        
        # Возвращаем результат
        return {
            "status": "completed"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при выполнении: {str(e)}"
        )

from fastapi import APIRouter, HTTPException, Depends

from app.api.schemas import AgentResponse, AgentDiscussionMetrics
from app.api.deps import get_agent_metrics_manager, AgentsResponseManager
from app.logs import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/agents", tags=["agents"])

@router.post(
    "/add",
    summary="Добавить метрики ответа агента",
    description="Сохраняет метрики использования (токены и т.п.) одного ответа агента",
    response_description="Идентификатор ответа и признак добавления",
)
async def add_agent_response(
    response: AgentResponse,
    manager: AgentsResponseManager = Depends(get_agent_metrics_manager)
):
    try:
        added = manager.add_response(response)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if not added:
        raise HTTPException(
            status_code=409,
            detail=f"Метрики ответа {response.response_id} уже существуют"
        )
    return {"response_id": response.response_id, "added": True}

@router.get(
    "/discussions/{discussion_id}",
    response_model=AgentDiscussionMetrics,
    summary="Получить метрики дискуссии",
    description="Возвращает метрики всех ответов агентов дискуссии и суммарную сводку по токенам",
    response_description="Метрики ответов дискуссии и их сводка",
)
async def get_discussion_metrics(
    discussion_id: str,
    manager: AgentsResponseManager = Depends(get_agent_metrics_manager)
):
    try:
        return manager.get_discussion_metrics(discussion_id)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get(
    "/{response_id}",
    response_model=AgentResponse,
    summary="Получить метрики ответа агента",
    description="Возвращает метрики указанного ответа агента по его идентификатору",
    response_description="Метрики ответа агента",
)
async def get_agent_response(
    response_id: str,
    manager: AgentsResponseManager = Depends(get_agent_metrics_manager)
):
    try:
        response = manager.get_response_by_id(response_id)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if response is None:
        raise HTTPException(status_code=404, detail=f"Метрики ответа {response_id} не найдены")
    return response

@router.delete(
    "/{response_id}",
    summary="Удалить метрики ответа агента",
    description="Удаляет сохранённые метрики указанного ответа агента из системы",
    response_description="Идентификатор ответа и признак удаления",
)
async def delete_agent_response(
    response_id: str,
    manager: AgentsResponseManager = Depends(get_agent_metrics_manager)
):
    try:
        deleted = manager.delete_response(response_id)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Метрики ответа {response_id} не найдены")
    return {"response_id": response_id, "deleted": True}

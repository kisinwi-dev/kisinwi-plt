from fastapi import APIRouter, HTTPException, Depends

from app.api.schemas import AgentResponse, AgentDiscussionMetrics
from app.api.deps import get_agent_metrics_manager, AgentsResponseManager
from app.logs import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/agents", tags=["agents"])

@router.post("/add")
async def add_agent_response(
    response: AgentResponse,
    manager: AgentsResponseManager = Depends(get_agent_metrics_manager)
):
    """
    Добавление нового ответа агента
    """
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

@router.get("/discussions/{discussion_id}", response_model=AgentDiscussionMetrics)
async def get_discussion_metrics(
    discussion_id: str,
    manager: AgentsResponseManager = Depends(get_agent_metrics_manager)
):
    """
    Метрики всех агентов дискуссии и суммарная сводка (токены и т.п.)
    """
    try:
        return manager.get_discussion_metrics(discussion_id)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{response_id}", response_model=AgentResponse)
async def get_agent_response(
    response_id: str,
    manager: AgentsResponseManager = Depends(get_agent_metrics_manager)
):
    """
    Получение ответа агента по ID
    """
    try:
        response = manager.get_response_by_id(response_id)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if response is None:
        raise HTTPException(status_code=404, detail=f"Метрики ответа {response_id} не найдены")
    return response

@router.delete("/{response_id}")
async def delete_agent_response(
    response_id: str,
    manager: AgentsResponseManager = Depends(get_agent_metrics_manager)
):
    """
    Удаление ответа агента
    """
    try:
        deleted = manager.delete_response(response_id)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Метрики ответа {response_id} не найдены")
    return {"response_id": response_id, "deleted": True}

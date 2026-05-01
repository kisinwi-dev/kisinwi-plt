from fastapi import APIRouter, HTTPException, Depends
from app.api.schemes import AgentResponse
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
        manager.add_response(response)
        return True
        
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
        
        return response
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{response_id}")
async def delete_agent_response(
    response_id: str,
    manager: AgentsResponseManager = Depends(get_agent_metrics_manager)
):
    """
    Удаление ответа агента
    """
    try:
        manager.delete_response(response_id)
        
        return True
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

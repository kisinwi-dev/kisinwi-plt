from fastapi import APIRouter, HTTPException, Depends

from app.api.schemas import AgentResponse
from app.api.deps import agent_response_manager
from app.logs import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/response", tags=["agents"])

@router.post("/add")
async def add_agent_response(
    response: AgentResponse,
):
    """
    Добавление нового ответа агента
    """
    try:
        # manager.add_response(response)
        return True
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{response_id}", response_model=AgentResponse)
async def get_agent_response(
    response_id: str,
):
    """
    Получение ответа агента по ID
    """
    try:
        pass
                
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{response_id}")
async def delete_agent_response(
    response_id: str,
):
    """
    Удаление ответа агента
    """
    try:
        pass
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

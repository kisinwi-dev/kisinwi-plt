from fastapi import APIRouter, HTTPException, Depends

from app.api.schemes import AgentResponse
from app.api.deps import agent_response_manager
from app.logs import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/discussion", tags=["agents"])

@router.get("/{discussion_id}")
async def get_agent_response(
    discussion_id: str
):
    """
    Получение ответов агентов в диалоге
    """
    try:
        pass

    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{discussion_id}")
async def delete_agent_response(
    discussion_id: str
):
    """
    Удаление диалога
    """
    try:
        pass

    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

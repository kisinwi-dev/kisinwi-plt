from fastapi import APIRouter, HTTPException, Response, status

from app.api.schemas import AgentResponseCreate, AgentResponse
from app.api.deps import agent_response_manager
from app.logs import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/response", tags=["agents"])

@router.post(
    "",
    status_code=201,
    summary="Сохранить ответ агента"
)
async def save_response(req: AgentResponseCreate):
    """Сохранить новый ответ агента"""
    try:
        
        agent_response_manager.save_response(
            discussion_id=req.discussion_id,
            response=req.agent_response
        )
        
        return Response(
            status_code=status.HTTP_201_CREATED
        )
    except Exception as e:
        logger.error(f"Ошибка при сохранении ответа агента: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при сохранении ответа агента: {str(e)}")

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

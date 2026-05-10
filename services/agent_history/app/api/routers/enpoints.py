from fastapi import APIRouter, HTTPException, Response, status

from app.api.schemas import AgentResponse, AgentResponseCreate
from app.api.deps import agent_response_manager
from app.logs import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["agents"])

@router.post(
    "/{discussion_id}/{response_id}",
    status_code=201,
    summary="Сохранить ответ агента"
)
async def save_response(
    response_id: str,
    discussion_id: str,
    req: AgentResponseCreate
):
    """Сохранить новый ответ агента"""
    try:
        
        agent_response_manager.save_response(
            discussion_id=discussion_id,
            response=AgentResponse(
                **req.model_dump(),
                response_id=response_id
            )
        )
        
        return Response(
            status_code=status.HTTP_201_CREATED
        )
    except Exception as e:
        logger.error(f"Ошибка при сохранении ответа агента: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при сохранении ответа агента: {str(e)}")

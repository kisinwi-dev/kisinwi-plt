from fastapi import APIRouter, HTTPException, Response, status

from app.api.schemas import AgentResponse, AgentResponseCreate
from app.api.deps import agent_response_manager
from app.logs import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["agents"])

@router.post(
    "/{discussion_id}/{response_id}",
    summary="Сохранить ответ агента",
    status_code=201,
    responses={
        201: {"description": "Ответ агента успешно сохранён"},
    }
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

@router.delete(
    "/{discussion_id}",
    summary="Удалить дискуссию",
    status_code=204,
    responses={
        204: {"description": "Дискуссия успешно удалена"},
        404: {"description": "Дискуссия не найдена"}
    }
)
async def delete_discussion(
    discussion_id: str
):
    """Сохранить новый ответ агента"""
    try:
        
        success = agent_response_manager.delete_discussion(
            discussion_id=discussion_id,
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Дискуссия '{discussion_id}' не найдена"
            )
        
        return Response(
            status_code=status.HTTP_204_NO_CONTENT
        )
    except Exception as e:
        logger.error(f"Ошибка при удалении дискуссии: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при удалении дискуссии: {str(e)}")

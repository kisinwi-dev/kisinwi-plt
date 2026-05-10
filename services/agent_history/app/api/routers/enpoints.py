from fastapi import APIRouter, HTTPException, Response, status

from app.api.schemas import AgentResponse, AgentResponseCreate, Discussion
from app.api.deps import agent_response_manager
from app.logs import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["agents"])

@router.get(
    "/discussions/{discussion_id}",
    summary="Получить все ответы агентов в дискуссии",
    response_model=Discussion,
    status_code=200,
    responses={
        200: {"description": "Ответы агентов"},
        404: {"description": "Ответы агентов или дискуссия не найдены"}
    }
)
async def get_response_in_discussion(
    discussion_id: str
):
    """Получить ответы агентов"""
    try:

        agents_responses = agent_response_manager.get_discussion_history(
            discussion_id=discussion_id
        )

        if agents_responses is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Дискуссия '{discussion_id}' не найдена"
            )

        return Discussion(
            discussion_id=discussion_id,
            agents_responses=agents_responses
        )
    except Exception as e:
        logger.error(f"Ошибка при удалении дискуссии: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при удалении дискуссии: {str(e)}")


@router.delete(
    "/discussions/{discussion_id}",
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

@router.post(
    "/discussions/{discussion_id}/responses/{response_id}",
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

from fastapi import APIRouter, HTTPException, Response, status

from app.api.schemas import AgentResponse
from app.api.deps import response_storage, discussion_stream_broker
from app.logs import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["agent response"])


@router.post(
    "/discussions/{discussion_id}/responses",
    summary="Сохранить ответ агента",
    status_code=201,
    responses={
        201: {"description": "Ответ агента успешно сохранён"},
    }
)
async def save_response(
    discussion_id: str,
    response: AgentResponse
):
    """Сохранить новый ответ агента"""
    try:
        await response_storage.save(
            discussion_id=discussion_id,
            response=response
        )
        discussion_stream_broker.publish(discussion_id)

        return Response(status_code=status.HTTP_201_CREATED)
    except Exception as e:
        logger.error(f"Ошибка при сохранении ответа агента: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при сохранении ответа агента: {str(e)}")


@router.get(
    "/discussions/{discussion_id}/responses",
    summary="Получить ответы агентов по дискуссии",
    response_model=list[AgentResponse],
    status_code=200,
    responses={
        200: {"description": "Список ответов агентов, отсортированный по времени"},
        404: {"description": "Дискуссия не найдена"},
    }
)
async def get_responses(discussion_id: str):
    """Получить все ответы агентов в рамках дискуссии"""
    try:
        responses = await response_storage.get_all(discussion_id=discussion_id)

        if responses is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Дискуссия '{discussion_id}' не найдена"
            )

        return responses
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при получении ответов агентов: {e}")
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import APIRouter, HTTPException, Response, status

from app.api.schemas import AgentResponse
from app.api.deps import response_storage
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

        response_storage.save(
            discussion_id=discussion_id,
            response=response
        )

        return Response(status_code=status.HTTP_201_CREATED)
    except Exception as e:
        logger.error(f"Ошибка при сохранении ответа агента: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при сохранении ответа агента: {str(e)}")

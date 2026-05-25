from fastapi import APIRouter, HTTPException, Response, status

from app.api.schemas import Discussion
from app.api.deps import agent_response_manager
from app.logs import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["discussion"])

@router.get(
    "/discussions/{discussion_id}",
    summary="Получить всю историю дискуссии",
    response_model=Discussion,
    status_code=200,
    responses={
        200: {"description": "Информация дискуссии"},
        404: {"description": "Ответы агентов или дискуссия не найдены"}
    }
)
async def get_full_discussion_messages(
    discussion_id: str
):
    """Получить историю дискусии"""
    try:

        events = agent_response_manager.get_discussion_history(
            discussion_id=discussion_id
        )

        if events is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Дискуссия '{discussion_id}' не найдена"
            )

        return Discussion(
            discussion_id=discussion_id,
            events=events
        )
    except HTTPException as e:
        logger.error(f"Ошибка при удалении дискуссии: {e}")
        raise e
    except Exception as e:
        logger.error(f"Ошибка при получении всей информации о дискуссии: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при получении всей информации о дискуссии: {str(e)}")


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
    """Удалить дискуссию"""
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
    except HTTPException as e:
        logger.error(f"Ошибка при удалении дискуссии: {e}")
        raise e
    except Exception as e:
        logger.error(f"Ошибка при удалении дискуссии: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка при удалении дискуссии: {str(e)}")

from fastapi import APIRouter, HTTPException, Response, status

from app.api.schemas import SystemMessage
from app.api.deps import system_storage
from app.logs import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["system"])


@router.post(
    "/discussions/{discussion_id}/system_messages",
    summary="Сохранение сообщения от системы",
    status_code=201,
    responses={
        201: {"description": "Сообщение системы успешно сохранено"},
    }
)
async def post_system_message(
    discussion_id: str,
    message: SystemMessage
):
    """Сохранить сообщение системы"""
    try:
        await system_storage.save(
            discussion_id=discussion_id,
            message=message
        )

        return Response(status_code=status.HTTP_201_CREATED)
    except Exception as e:
        logger.error(f"Ошибка при сохранении сообщения от системы: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка при сохранении сообщения от системы: {str(e)}")


@router.get(
    "/discussions/{discussion_id}/system_messages",
    summary="Получить системные сообщения дискуссии",
    response_model=list[SystemMessage],
    status_code=200,
    responses={
        200: {"description": "Список системных сообщений, отсортированный по времени"},
        404: {"description": "Дискуссия не найдена"},
    }
)
async def get_system_messages(discussion_id: str):
    """Получить все системные сообщения в рамках дискуссии"""
    try:
        messages = await system_storage.get_all(discussion_id=discussion_id)

        if messages is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Дискуссия '{discussion_id}' не найдена"
            )

        return messages
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при получении системных сообщений: {e}")
        raise HTTPException(status_code=500, detail=str(e))

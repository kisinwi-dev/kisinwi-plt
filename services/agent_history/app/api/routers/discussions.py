from fastapi import APIRouter, HTTPException, Response, status

from app.api.schemas import Discussion, DiscussionMeta, DiscussionMetaUpdate, CreateDiscussion
from app.api.deps import discussion_storage
from app.logs import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["discussion"])


@router.post(
    "/discussions",
    summary="Создать дискуссию",
    response_model=DiscussionMeta,
    status_code=201,
    responses={
        201: {"description": "Дискуссия успешно создана"},
    }
)
async def create_discussion(data: CreateDiscussion):
    """Создать новую дискуссию с метаданными"""
    try:
        meta = discussion_storage.create(data)
        return meta
    except Exception as e:
        logger.error(f"Ошибка при создании дискуссии: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/discussions",
    summary="Список всех дискуссий",
    response_model=list[DiscussionMeta],
    status_code=200,
    responses={
        200: {"description": "Список дискуссий с метаданными"},
    }
)
async def list_discussions():
    """Получить список всех дискуссий"""
    try:
        return discussion_storage.get_all()
    except Exception as e:
        logger.error(f"Ошибка при получении списка дискуссий: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/discussions/{discussion_id}",
    summary="Получить всю историю дискуссии",
    response_model=Discussion,
    status_code=200,
    responses={
        200: {"description": "Информация дискуссии"},
        404: {"description": "Дискуссия не найдена"}
    }
)
async def get_full_discussion_messages(discussion_id: str):
    """Получить историю дискуссии"""
    try:
        events = discussion_storage.get_history(discussion_id=discussion_id)

        if events is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Дискуссия '{discussion_id}' не найдена"
            )

        meta = discussion_storage.get_meta(discussion_id)

        return Discussion(
            discussion_id=discussion_id,
            meta=meta,
            events=events
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при получении всей информации о дискуссии: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch(
    "/discussions/{discussion_id}/meta",
    summary="Обновить метаданные дискуссии",
    response_model=DiscussionMeta,
    status_code=200,
    responses={
        200: {"description": "Метаданные обновлены"},
        404: {"description": "Дискуссия не найдена"}
    }
)
async def update_discussion_meta(discussion_id: str, update: DiscussionMetaUpdate):
    """Обновить метаданные дискуссии"""
    try:
        meta = discussion_storage.update_meta(discussion_id=discussion_id, update=update)

        if meta is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Метаданные дискуссии '{discussion_id}' не найдены"
            )

        return meta
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при обновлении метаданных: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/discussions/{discussion_id}",
    summary="Удалить дискуссию",
    status_code=204,
    responses={
        204: {"description": "Дискуссия успешно удалена"},
        404: {"description": "Дискуссия не найдена"}
    }
)
async def delete_discussion(discussion_id: str):
    """Удалить дискуссию"""
    try:
        success = discussion_storage.delete(discussion_id=discussion_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Дискуссия '{discussion_id}' не найдена"
            )

        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при удалении дискуссии: {e}")
        raise HTTPException(status_code=500, detail=str(e))

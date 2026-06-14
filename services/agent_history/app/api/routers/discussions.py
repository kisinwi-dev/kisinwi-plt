import asyncio
import json
from typing import Optional
from fastapi import APIRouter, HTTPException, Query, Response, status
from fastapi.responses import StreamingResponse

from app.api.schemas import (
    DiscussionMeta,
    DiscussionMetaRead,
    DiscussionMetaUpdate,
    CreateDiscussion,
    DiscussionStatus,
    DiscussionSnapshot,
)
from app.api.deps import (
    discussion_storage,
    response_storage,
    system_storage,
    discussion_stream_broker,
)
from app.core.stream import format_sse
from app.logs import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["discussion"])

# Интервал keepalive-комментариев SSE, чтобы соединение не закрывали прокси
KEEPALIVE_INTERVAL_S = 15.0

# Статусы, после которых дискуссия не обновляется
FINAL_DISCUSSION_STATUSES = (
    DiscussionStatus.COMPLETED, DiscussionStatus.FAILED, DiscussionStatus.CANCELLED
)


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
        meta = await discussion_storage.create(data)
        return meta
    except Exception as e:
        logger.error(f"Ошибка при создании дискуссии: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/discussions",
    summary="Список всех дискуссий",
    response_model=list[DiscussionMetaRead],
    status_code=200,
    responses={
        200: {"description": "Список дискуссий с метаданными"},
    }
)
async def list_discussions(
    status_filter: Optional[DiscussionStatus] = Query(None, alias="status"),
    pipeline: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
):
    """Получить список всех дискуссий"""
    try:
        return await discussion_storage.get_all(
            status=status_filter,
            pipeline=pipeline,
            skip=skip,
            limit=limit,
        )
    except Exception as e:
        logger.error(f"Ошибка при получении списка дискуссий: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/discussions/{discussion_id}/meta",
    summary="Получить метаданные дискуссии",
    response_model=DiscussionMeta,
    status_code=200,
    responses={
        200: {"description": "Метаданные дискуссии"},
        404: {"description": "Дискуссия не найдена"},
    }
)
async def get_discussion_meta(discussion_id: str):
    """Получить метаданные одной дискуссии"""
    try:
        meta = await discussion_storage.get_meta(discussion_id=discussion_id)

        if meta is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Дискуссия '{discussion_id}' не найдена"
            )

        return meta
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при получении метаданных дискуссии: {e}")
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
        meta = await discussion_storage.update_meta(discussion_id=discussion_id, update=update)

        if meta is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Метаданные дискуссии '{discussion_id}' не найдены"
            )

        # Смена статуса на completed/failed доставит подписчикам SSE событие end
        discussion_stream_broker.publish(discussion_id)
        return meta
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при обновлении метаданных: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/discussions/{discussion_id}/stream",
    summary="Поток дискуссии (SSE)",
    description="Server-Sent Events: при подключении отправляет событие discussion с текущим "
                "снимком дискуссии (схема DiscussionSnapshot: meta, ответы агентов и системные "
                "сообщения, отсортированные по времени), затем шлёт такое же событие при каждом "
                "новом ответе, системном сообщении или обновлении метаданных. "
                "Когда статус дискуссии становится финальным (completed/failed), после снимка "
                "отправляется событие end с телом {discussion_id, status} — один раз на соединение; "
                "подписчику уже завершённой дискуссии снимок и end приходят сразу при подключении. "
                "После end поток не закрывается сервером — клиент закрывает EventSource сам. "
                "Каждые 15 секунд отправляется keepalive-комментарий. "
                "Для несуществующей дискуссии отдаётся снимок с meta=null — "
                "подключаться можно до её создания",
    response_description="Поток событий text/event-stream",
    responses={
        200: {
            "description": "Поток SSE: события discussion с JSON-телом схемы DiscussionSnapshot "
                           "и событие end по завершении дискуссии",
            "content": {
                "text/event-stream": {
                    "schema": {"type": "string"},
                    "example": 'event: discussion\n'
                               'data: {"discussion_id": "d-1", "meta": {"discussion_id": "d-1", '
                               '"status": "completed", "pipeline": "development"}, '
                               '"responses": [{"response_id": "r-1", "agent_role": "analyst", '
                               '"text": "..."}], "system_messages": []}\n\n'
                               'event: end\n'
                               'data: {"discussion_id": "d-1", "status": "completed"}\n\n',
                }
            },
        },
    },
)
async def stream_discussion(discussion_id: str):
    """SSE-поток ленты дискуссии: снимок при подключении и после каждого обновления"""

    async def snapshot() -> DiscussionSnapshot:
        meta = await discussion_storage.get_meta(discussion_id=discussion_id)
        responses = await response_storage.get_all(discussion_id=discussion_id)
        messages = await system_storage.get_all(discussion_id=discussion_id)
        return DiscussionSnapshot(
            discussion_id=discussion_id,
            meta=meta,
            responses=responses or [],
            system_messages=messages or [],
        )

    async def event_generator():
        queue = discussion_stream_broker.subscribe(discussion_id)
        end_sent = False

        async def frames() -> str:
            """Кадр снимка + однократный кадр end при финальном статусе"""
            nonlocal end_sent
            snap = await snapshot()
            out = format_sse("discussion", snap.model_dump_json())
            if (
                not end_sent
                and snap.meta is not None
                and snap.meta.status in FINAL_DISCUSSION_STATUSES
            ):
                out += format_sse(
                    "end",
                    json.dumps({
                        "discussion_id": discussion_id,
                        "status": snap.meta.status.value,
                    }),
                )
                end_sent = True
            return out

        try:
            yield await frames()
            # После end поток не закрываем: текущие клиенты EventSource
            # иначе уйдут в бесконечный цикл переподключений
            while True:
                try:
                    await asyncio.wait_for(queue.get(), timeout=KEEPALIVE_INTERVAL_S)
                    # Схлопываем накопившиеся уведомления — снимок всё равно полный
                    while not queue.empty():
                        queue.get_nowait()
                    yield await frames()
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        finally:
            discussion_stream_broker.unsubscribe(discussion_id, queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


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
        success = await discussion_storage.delete(discussion_id=discussion_id)

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

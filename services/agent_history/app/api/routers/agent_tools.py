from fastapi import APIRouter, HTTPException, Response, status

from app.api.schemas import Tool
from app.api.deps import tool_storage
from app.logs import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["agent tools"])


@router.post(
    "/discussions/{discussion_id}/tool",
    summary="Сохранение информации об инструменте",
    status_code=201,
    responses={
        201: {"description": "Информацию об инструменте успешно сохранено"},
    }
)
async def post_tool(
    discussion_id: str,
    tool_info: Tool
):
    """Сохранить информацию об инструменте"""
    try:
        tool_storage.save(
            discussion_id=discussion_id,
            tool=tool_info
        )

        return Response(status_code=status.HTTP_201_CREATED)
    except Exception as e:
        logger.error(f"Ошибка при сохранении информации об инструменте: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка при сохранении информации об инструменте: {str(e)}")


@router.get(
    "/discussions/{discussion_id}/responses/{response_id}/tools",
    summary="Получить инструменты конкретного запуска агента",
    response_model=list[Tool],
    status_code=200,
    responses={
        200: {"description": "Список инструментов запуска агента"},
        404: {"description": "Дискуссия не найдена"},
    }
)
async def get_tools_by_response(
    discussion_id: str,
    response_id: str,
):
    """Получить все инструменты, вызванные в рамках конкретного запуска агента (по response_id)"""
    try:
        tools = tool_storage.get_by_response(
            discussion_id=discussion_id,
            response_id=response_id,
        )

        if tools is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Дискуссия '{discussion_id}' не найдена",
            )

        return tools

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка при получении инструментов по response_id: {e}")
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import APIRouter, HTTPException, Response, status

from app.api.schemas import Tool
from app.api.deps import agent_response_manager
from app.logs import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["tools"])

@router.post(
    "/discussions/{discussion_id}/tool",
    summary="Сохранение информации об инструменте",
    status_code=201,
    responses={
        201: {"description": "Информацию об инструмене успешно сохранено"},
    }
)
async def post_tool(
    discussion_id: str,
    tool_info: Tool
):
    """Сохранить информацию об инструменте"""
    try:

        agent_response_manager.save_tool_info(
            discussion_id=discussion_id,
            tool_info=tool_info
        )

        return Response(
            status_code=status.HTTP_201_CREATED
        )
    except Exception as e:
        logger.error(f"Ошибка при сохранении информации об инструементе: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка при сохранении информации об инструменте: {str(e)}")

from typing import Optional
from pydantic import BaseModel, Field

from .agent import AgentResponse
from .discussion_meta import DiscussionMeta
from .system_message import SystemMessage


class DiscussionSnapshot(BaseModel):
    """Полный снимок дискуссии для SSE-потока"""
    discussion_id: str = Field(..., description="ID дискуссии")
    meta: Optional[DiscussionMeta] = Field(
        None,
        description="Метаданные дискуссии; None — дискуссия ещё не создана",
    )
    responses: list[AgentResponse] = Field(
        default_factory=list,
        description="Ответы агентов, отсортированные по времени",
    )
    system_messages: list[SystemMessage] = Field(
        default_factory=list,
        description="Системные сообщения, отсортированные по времени",
    )

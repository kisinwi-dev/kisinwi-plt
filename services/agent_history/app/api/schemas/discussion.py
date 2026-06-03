from typing import List, Optional
from pydantic import BaseModel, Field

from .discussion_meta import DiscussionMeta


class Discussion(BaseModel):
    discussion_id: str = Field(..., description="ID дискуссии")
    meta: Optional[DiscussionMeta] = Field(None, description="Метаданные дискуссии")
    events: List[dict] = Field(..., description="Событие. Сообщение от агента/системы/инструмента")

from typing import List
from pydantic import BaseModel, Field

class Discussion(BaseModel):
    discussion_id: str = Field(..., description="ID дискуссии")
    events: List[dict] = Field(..., description="Событие. Сообщение от агента или от системы")

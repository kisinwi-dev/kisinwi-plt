from typing import List
from pydantic import BaseModel, Field

from .agent import AgentResponse

class Discussion(BaseModel):
    discussion_id: str = Field(..., description="ID дискуссии")
    agents_responses: List[AgentResponse] = Field(..., description="Ответы агентов")

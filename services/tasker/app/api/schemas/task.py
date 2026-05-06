from typing import List
from pydantic import BaseModel, Field

class TaskCreate(BaseModel):
    task_name: str
    model_id: str
    discussion_id: str | None = Field(None, description="Id диалога к которому относится задача")

class TaskUpdate(BaseModel):
    status: str
    status_info: str
    error: str | None = None

class AddAgentResponse(BaseModel):
    task_id: str
    agent_response_id: str

class StatusesInfo(BaseModel):
    statuses: List[Status] = Field(..., description="Список статусов")

class Status(BaseModel):
    id: int = Field(..., description="Id статуса")
    status: str = Field(..., description="Название статуса")
    description: str = Field(..., description="Описание статуса")

class TaskStatistics(BaseModel):
    count: int = Field(..., description="Количество задач")
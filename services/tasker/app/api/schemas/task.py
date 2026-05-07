from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field

class TasksResponse(BaseModel):
    tasks: List[TaskResponse] = Field(..., description="Список задач с полной информацией о них")

class TaskResponse(BaseModel):
    id: str
    name: str
    model_id: str
    discussion_id: Optional[str]
    agent_respons_ids: List[str]
    status_id: int
    status: str
    status_description: str
    percentages: int
    status_info: Optional[str]
    error_message: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    updated_at: Optional[datetime]
    completed_at: Optional[datetime]

class TaskResponseMin(BaseModel):
    id: str
    model_id: str

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
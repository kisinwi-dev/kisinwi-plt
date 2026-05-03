from pydantic import BaseModel 

class TaskCreate(BaseModel):
    task_name: str
    model_id: str
    discussion_id: str | None

class TaskUpdate(BaseModel):
    status: str
    status_info: str
    error: str | None = None

class AddAgentResponse(BaseModel):
    task_id: str
    agent_response_id: str
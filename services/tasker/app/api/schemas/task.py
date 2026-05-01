from pydantic import BaseModel 

class TaskCreate(BaseModel):
    model_id: str
    discussion_id: str | None

class TaskUpdate(BaseModel):
    task_id: str
    status: str
    error: str | None

class AddAgentResponse(BaseModel):
    task_id: str
    agent_response_id: str
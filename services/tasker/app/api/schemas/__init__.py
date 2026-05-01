from .info import HealthResponse

from typing import Optional
from pydantic import BaseModel 

class TaskCreate(BaseModel):
    params: dict

class TaskUpdate(BaseModel):
    status: Optional[str] = None
    progress: Optional[int] = None
    result: Optional[dict] = None
    description: Optional[str] = None
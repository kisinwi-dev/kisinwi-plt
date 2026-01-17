from pydantic import BaseModel, Field
from typing import List, Optional

class ClassInfoResponse(BaseModel):
    name: str
    description: Optional[str] = None
    count_files: Optional[int] = None
    type_files: Optional[str] = None

class ClassListResponse(BaseModel):
    dataset_name: str
    version_name: str
    classes: List[str]

class ClassRenameRequest(BaseModel):
    new_name: str

class ClassCreateRequest(BaseModel):
    class_name: str
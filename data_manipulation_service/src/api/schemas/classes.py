from pydantic import BaseModel, Field
from typing import List, Optional

class ClassSchema(BaseModel):
    name: str = Field(..., example="cat")
    description: Optional[str] = None
    count_files: int = Field(..., example=120)
    type_files: Optional[str] = Field(None, example="image")

    class Config:
        from_attributes = True

class ClassListResponse(BaseModel):
    dataset_name: str
    version_name: str
    classes: List[str]

class ClassRenameRequest(BaseModel):
    new_name: str

class ClassCreateRequest(BaseModel):
    class_name: str
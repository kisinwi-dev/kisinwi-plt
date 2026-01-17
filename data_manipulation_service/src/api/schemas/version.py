from pydantic import BaseModel, Field
from typing import List, Optional
from .classes import ClassSchema

class VersionSchema(BaseModel):
    name: str = Field(..., example="v_0")
    description: Optional[str] = None
    classes: List[ClassSchema]

    class Config:
        from_attributes = True

class VersionCreateRequest(BaseModel):
    version_name: str

class VersionRenameRequest(BaseModel):
    new_name: str

class VersionListResponse(BaseModel):
    versions: list[str]
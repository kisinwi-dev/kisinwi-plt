from pydantic import BaseModel, Field
from typing import List, Optional
from .classes import ClassInfoResponse

class VersionInfoResponse(BaseModel):
    name: str
    description: Optional[str] = None
    classes: List[ClassInfoResponse]

class VersionCreateRequest(BaseModel):
    version_name: str

class VersionRenameRequest(BaseModel):
    new_name: str

class VersionListResponse(BaseModel):
    versions: list[str]
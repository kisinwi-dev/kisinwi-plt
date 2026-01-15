from pydantic import BaseModel, Field
from typing import List, Optional
from .version import VersionSchema

class DatasetSchema(BaseModel):
    name: str = Field(..., example="animals")
    description: Optional[str] = None
    versions: List[VersionSchema]

    class Config:
        from_attributes = True

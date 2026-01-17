from pydantic import BaseModel, Field
from typing import List, Optional

class ClassSchema(BaseModel):
    name: str = Field(..., example="cat")
    description: Optional[str] = None
    count_files: int = Field(..., example=120)
    type_files: Optional[str] = Field(None, example="image")

    class Config:
        from_attributes = True
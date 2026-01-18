from fastapi import Path
from pydantic import BaseModel, Field
from typing import List, Optional

CLASS_NAME_PATH = Path(
    ..., 
    description="Unique name of the class", 
    example="class_1",
    pattern=r"^[\w\-]+$"
)

# ------------------ Request management ------------------

class ClassRenameRequest(BaseModel):
    new_name: str = Field(
        ..., 
        example="dog", 
        description="New name for the class"
    )

class ClassCreateRequest(BaseModel):
    class_name: str = Field(
        ..., 
        example="cat", 
        description="Name of the new class"
    )

# ------------------ Response management ------------------

class ClassInfoResponse(BaseModel):
    name: str = Field(
        ..., 
        example="cat", 
        description="Name of the class"
    )
    description: Optional[str] = Field(
        None, 
        example="Images of cats", 
        description="Optional description of the class"
    )
    count_files: Optional[int] = Field(
        None, 
        example=100, 
        description="Number of files in the class"
    )
    type_files: Optional[str] = Field(
        None, 
        example="jpg", 
        description="Type of files in the class"
    )

class ClassListResponse(BaseModel):
    classes: List[str] = Field(
        ..., 
        example=["cat", "dog", "deer"], 
        description="List of class names in the dataset version"
    )

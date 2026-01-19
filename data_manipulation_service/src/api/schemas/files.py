from fastapi import Path
from pydantic import BaseModel, Field
from typing import List

FILE_NAME_PATH = Path(
    ...,
    description="Unique name of the file",
    examples=["dog_1", "dog_2"],
    pattern=r"^[\w\-]+$"
)

# ------------------ Request management ------------------


# ------------------ Response management ------------------

class FileListResponse(BaseModel):
    files: List[str] = Field(
        ...,
        example=["dog_1.jpg", "dog_2.jpg", "dog_3.jpg"],
        description="List of file names in the class of the dataset version"
    )

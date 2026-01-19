from fastapi import Path
from pydantic import BaseModel, Field
from typing import List, Optional
from .classes import ClassInfoResponse

VERSION_NAME_PATH = Path(
    ...,
    description="Unique name of the version",
    examples=["v_1", "v_2"],
    pattern=r"^[\w\-]+$"
)

# ------------------ Request management ------------------


class VersionCreateRequest(BaseModel):
    version_name: str = Field(
        ...,
        example="v_1",
        description="Name of the new dataset version",
        pattern=r"^[\w\-]+$"
    )


class VersionRenameRequest(BaseModel):
    new_name: str = Field(
        ...,
        example="v_2",
        description="New name for the version"
    )

# ------------------ Response management ------------------


class VersionInfoResponse(BaseModel):
    name: str = Field(
        ...,
        example="v_1",
        description="Version name of the dataset"
    )
    description: Optional[str] = Field(
        None,
        example="First version of dataset",
        description="Optional description of the version"
    )
    classes: List[ClassInfoResponse] = Field(
        ...,
        description="List of classes in this version"
    )


class VersionListResponse(BaseModel):
    versions: List[str] = Field(
        ...,
        example=["v_1", "v_2"],
        description="List of dataset version names"
    )

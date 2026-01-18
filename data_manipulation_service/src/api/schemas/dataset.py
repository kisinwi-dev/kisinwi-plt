from fastapi import Path
from pydantic import BaseModel, Field
from typing import List, Optional
from .version import VersionInfoResponse

DATASET_NAME_PATH = Path(
    ...,
    description="Unique name of the dataset",
    example="dataset_1",
    pattern=r"^[\w\-]+$"
)

# ------------------ Request management ------------------


class DatasetCreateRequest(BaseModel):
    dataset_name: str = Field(
        ...,
        example="dataset_1",
        description="Name of the dataset to create"
    )
    archive_name: str = Field(
        ...,
        example="dataset_1.zip",
        description="Name of the archive containing dataset files"
    )
    dataset_type: str = Field(
        ...,
        example="image",
        description="Type of the dataset"
    )
    dataset_task: str = Field(
        ...,
        example="classification",
        description="Task type for the dataset"
    )


class DatasetRenameRequest(BaseModel):
    new_name: str = Field(
        ...,
        example="dataset_new",
        description="New name for the dataset"
    )

# ------------------ Response management ------------------


class DatasetInfoResponse(BaseModel):
    name: str = Field(
        ...,
        example="dataset_1",
        description="Name of the dataset"
    )
    description: Optional[str] = Field(
        None,
        example="Dataset for image classification",
        description="Optional description of the dataset"
    )
    versions: List[VersionInfoResponse] = Field(
        ...,
        description="List of dataset versions"
    )


class DatasetListResponse(BaseModel):
    datasets: List[str] = Field(
        ...,
        example=["dataset_1", "dataset_2"],
        description="List of available dataset names"
    )

from datetime import datetime
from typing import Dict, List, Literal
from pydantic import BaseModel, Field, HttpUrl

class SourceItem(BaseModel):
    url: HttpUrl
    description: str

class Source(BaseModel):
    kaggle: SourceItem

class Version(BaseModel):
    version_id: str
    description: str
    size_bytes: int = Field(..., ge=0)
    num_samples: int = Field(..., ge=0)
    num_train: int = Field(..., ge=0)
    num_val: int = Field(..., ge=0)
    num_test: int = Field(..., ge=0)

    created_at: datetime = Field(default_factory=datetime.now, frozen=True)

class DatasetMetadata(BaseModel):
    dataset_id: str = Field(..., min_length=1)
    name: str
    description: str
    num_classes: int = Field(..., ge=1)
    class_names: List[str] = Field(..., min_length=1)
    class_to_idx: Dict[str, int]
    source: Source
    type: Literal["image", "text", "tabular", "other"] = "other"
    task: Literal["classification", "regression", "detection", "segmentation", "other"]
    default_version_id: str
    versions: List[Version] = Field(..., min_length=1)

    created_at: datetime = Field(default_factory=datetime.now, frozen=True)
    updated_at: datetime = Field(default_factory=datetime.now, frozen=True)

    model_config = {
        "validate_assignment": True
    }

    def __setattr__(self, name: str, value) -> None:
        super().__setattr__(name, value)
        object.__setattr__(self, "updated_at", datetime.now())

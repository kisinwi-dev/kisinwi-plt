from typing import Literal, List
from pydantic import BaseModel, Field, field_validator

from .dataset import Source

class NewVersion(BaseModel):
    version_id: str
    description: str

class NewDataset(BaseModel):
    dataset_id: str = Field(..., min_length=1)
    name: str
    description: str
    class_names: List[str] = Field(..., min_length=1)
    sources: List[Source]
    type: Literal["image", "text", "tabular", "other"] = "other"
    task: Literal["classification", "regression", "detection", "segmentation", "other"]
    version: NewVersion

    model_config = {
        "validate_assignment": True
    }

    @field_validator("class_names")
    @classmethod
    def validate_class_names(cls, v: List[str]):

        cleaned = [c.strip() for c in v]

        if any(c == "" for c in cleaned):
            raise ValueError("class_names не должен содержать путое поле")

        if len(cleaned) != len(set(cleaned)):
            raise ValueError("class_names должны быть уникальными")

        return cleaned
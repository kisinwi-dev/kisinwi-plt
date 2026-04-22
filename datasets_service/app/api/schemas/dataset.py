from datetime import datetime
from typing import Dict, List, Literal
from pydantic import (
    BaseModel, Field,
    HttpUrl, model_validator
)

class Source(BaseModel):
    type: Literal['kaggle', 'url', 'huggingface', 'other']
    url: HttpUrl
    description: str

class Version(BaseModel):
    version_id: str
    description: str
    size_bytes: int = Field(..., ge=0)
    num_samples: int = Field(..., ge=0)
    num_train: int = Field(..., ge=0)
    num_val: int = Field(..., ge=0)
    num_test: int = Field(..., ge=0)

    created_at: datetime = Field(default_factory=datetime.now, frozen=True)

    @model_validator(mode="after")
    def check_split_consistency(self):
        if self.num_train + self.num_val + self.num_test != self.num_samples:
            raise ValueError(
                "num_train + num_val + num_test != num_samples"
            )
        return self

class DatasetMetadata(BaseModel):
    dataset_id: str = Field(..., min_length=1)
    name: str
    description: str
    num_classes: int = Field(..., ge=1)
    class_names: List[str] = Field(..., min_length=1)
    class_to_idx: Dict[str, int]
    sources: List[Source]
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

    @model_validator(mode="after")
    def validate_class_mapping(self):

        if set(self.class_names) != set(self.class_to_idx.keys()):
            raise ValueError(
                "class_to_idx ключи должны совпадать с именами классов"
            )

        if len(self.class_names) != self.num_classes:
            raise ValueError(
                "num_classes должно быть равно количеству class_names"
            )

        return self
    
    @model_validator(mode="after")
    def validate_default_version(self):

        version_ids = [v.version_id for v in self.versions]

        if self.default_version_id not in version_ids:
            raise ValueError(
                "default_version_id должен существовать в версиях"
            )

        return self
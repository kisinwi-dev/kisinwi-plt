from pydantic import BaseModel, Field
from typing import List, Optional
from .version import VersionInfoResponse

class DatasetInfoResponse(BaseModel):
    name: str
    description: Optional[str] = None
    versions: List[VersionInfoResponse]

class DatasetCreateRequest(BaseModel):
    dataset_name: str
    archive_name: str
    dataset_type: str
    dataset_task: str

class DatasetListResponse(BaseModel):
    datasets: list[str]

class DatasetRenameRequest(BaseModel):
    new_name: str
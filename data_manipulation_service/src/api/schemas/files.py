from pydantic import BaseModel
from typing import List

class FileListResponse(BaseModel):
    dataset_name: str
    version_name: str
    class_name: str
    files: List[str]

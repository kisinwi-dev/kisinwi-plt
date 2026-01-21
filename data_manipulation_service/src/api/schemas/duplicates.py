from pydantic import BaseModel
from typing import List

class DuplicateFilesGroupResponse(BaseModel):
    duplicates: List[List[str]]

from typing import List, Optional
from pydantic import BaseModel, Field

class VersionFilesResponse(BaseModel):
    """Страница списка файлов версии"""
    dataset_id: str = Field(description="ID датасета")
    version_id: str = Field(description="ID версии")
    split: Optional[str] = Field(default=None, description="Сплит, по которому отфильтрован список")
    total: int = Field(ge=0, description="Всего файлов (с учётом фильтра по сплиту)")
    files: List[str] = Field(default_factory=list, description="Относительные пути файлов (split/class/filename)")

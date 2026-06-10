from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

class File(BaseModel):
    id: str = Field(..., description="ID файла")
    version_id: str = Field(..., description="ID версии модели, к которой относится файл")
    filename: str = Field(..., description="Имя файла")
    file_size: int = Field(..., description="Размер файла в байтах")
    created_at: datetime = Field(..., description="Создание файла")

class Files(BaseModel):
    files: List[File] = Field(..., description="Список файлов")

class FileDeletes(BaseModel):
    ids: Optional[List[str]] = Field(..., description="Список id файлов, которые нужно удалить")

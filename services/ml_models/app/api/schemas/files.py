from typing import List
from datetime import datetime
from pydantic import BaseModel, Field

class Files(BaseModel):
    files: List[File] = Field(..., description="Список файлов")

class File(BaseModel):
    id: str = Field(..., description="ID файла")
    model_id: str = Field(..., description="ID модели, к которой относится файл")
    filename: str = Field(..., description="Имя файла")
    file_size: int = Field(..., description="Размер файла в байтах")
    created_at: datetime = Field(..., description="Создание модели")
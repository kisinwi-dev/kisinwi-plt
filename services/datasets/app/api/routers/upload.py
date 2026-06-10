import os
import shutil
from pathlib import Path
from fastapi import UploadFile, APIRouter, File, Form, HTTPException

from app.logs import get_logger
from app.core.filesystem import ArchiveManager

logger = get_logger(__name__)
router = APIRouter(tags=["Upload"])

@router.post(
    "/upload",
    response_model=bool,
    summary="Загрузка данных",
    description="""
Загружает архив с данными и распаковывает во временную директорию. 
Идентификатор id_data связывает загруженные файлы с будущим датасетом.

- Данные временно хранятся до вызова */new
- Если id_data уже существует, будет вызвана ошибка
""",
    response_description="True, если данные успешно загружены",
)
def uploads_data(
    id_data: str = Form(default='my_data', description="Уникальный идентификатор данных"),
    file: UploadFile = File(..., description="Архив датасета"),
):
    af = ArchiveManager()
    save_path = None
    try:
        # сохраняем полное расширение, чтобы не терять составные (.tar.gz)
        file_suffixes = ''.join(Path(str(file.filename)).suffixes)
        save_path = af.save_file(file, f"{id_data}{file_suffixes}")
        
        af.unpack(save_path, id_data)
        return True
    except Exception as e:
        logger.error(f"Ошибка обработки загрузки `{id_data}`: {e}")
        if save_path and os.path.exists(save_path):
            os.remove(save_path)

        extracted_folder = af.temp_folder / id_data
        if extracted_folder.exists():
            shutil.rmtree(extracted_folder)

        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")
    
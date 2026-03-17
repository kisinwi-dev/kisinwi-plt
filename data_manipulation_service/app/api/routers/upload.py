from fastapi import UploadFile, APIRouter, File

from app.logs import get_logger
from app.core.filesystem import ArchiveManager

logger = get_logger(__name__)
router = APIRouter(tags=["Upload"])

@router.post(
    "/upload",
    response_model=bool,
    summary="Загрузка данных",
    response_description="True, если данные успешно загружены",
)
def uploads_data(
    id_data: str,
    file: UploadFile = File(..., description="Файл датасета"),
):
    af = ArchiveManager()
    save_path = af.save_file(file, f"{id_data}.{str(file.filename)[0]}")

    _ = af.unpack(save_path, id_data)
    return True
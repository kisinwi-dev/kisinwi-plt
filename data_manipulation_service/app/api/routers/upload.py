from fastapi import UploadFile, APIRouter, File

from app.logs import get_logger
from app.core.filesystem import ArchiveManager

logger = get_logger(__name__)
router = APIRouter(prefix="/datasets/{dataset_id}", tags=["Version"])

@router.post(
    "/upload",
    response_model=bool,
    summary="Загрузка данных",
    response_description="True, если данные успешно загружены",
)
def uploads_data(
    file: UploadFile = File(..., description="Файл датасета"),
):
    af = ArchiveManager()
    save_path = af.save_file(file, str(file.filename))

    _ = af.unpack(save_path, str(file.filename).split('.')[0])
    return True
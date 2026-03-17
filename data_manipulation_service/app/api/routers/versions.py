from fastapi import APIRouter, Depends, UploadFile, File
from typing import List

from app.logs import get_logger
from app.core.filesystem import ArchiveManager
from app.core.services import DatasetManager
from app.api.deps import get_dataset_manager
from app.api.schemas.dataset import Version
from app.api.schemas.dataset_new import NewVersion

logger = get_logger(__name__)
router = APIRouter(prefix="/datasets/{dataset_id}/versions", tags=["Version"])

@router.get(
    "/", 
    response_model=List[Version],
    summary="Получение списка версий",
    description="Возвращает список всех доступных версий датасета",
    response_description="Список версий",
)
def list_versions(
        dataset_id: str,
        dm: DatasetManager = Depends(get_dataset_manager)
):
    return dm.get_dataset_info(dataset_id).versions

@router.delete(
    "/{version_id}", 
    response_model=bool,
    summary="Удалить версию",
    description="Удаляет указанную версию датасета из системы",
    response_description="True, если версия была успешно удалёна",
)
def delete_version(
        dataset_id: str,
        version_id: str,
        dm: DatasetManager = Depends(get_dataset_manager)
):
    return dm.drop_version(dataset_id, version_id)

@router.post(
    "/new", 
    response_model=bool,
    summary="Добавить новую версию данных",
    description="Создаёт новую версию датасета из загруженных данных",
    response_description="True, если датасет успешно создан",
)
def create_version(
    dataset_id: str,
    new_dataset: NewVersion, 
    dm: DatasetManager = Depends(get_dataset_manager),
):
    return dm.add_new_version(dataset_id, new_dataset)

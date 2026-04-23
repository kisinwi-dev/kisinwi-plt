from fastapi import APIRouter, Depends, HTTPException
from typing import List

from app.logs import get_logger
from app.core.services import DatasetManager
from app.core.exception.base import CoreException
from app.api.deps import get_dataset_manager
from app.api.schemas.dataset import DatasetMetadata
from app.api.schemas.dataset_new import NewDataset

logger = get_logger(__name__)
router = APIRouter(prefix="/datasets", tags=["Datasets"])

@router.get(
    "/", 
    response_model=List[DatasetMetadata],
    summary="Получение списка датасетов",
    description="Возвращает список идентифакторов всех доступных датасето",
    response_description="Список идентификатор",
)
def list_datasets(dm: DatasetManager = Depends(get_dataset_manager)):
    ids = dm.get_datasets_id()
    dsms = []
    for id in ids:
        dsms.append(dm.get_dataset_info(id))
    return dsms

@router.get(
    "/{dataset_id}", 
    response_model=DatasetMetadata,
    summary="Получить метаданные датасета",
    description="Возвращает метаданные указанного датасета по его идентификатору",
    response_description="Метаданные датасета",
)
def get_dataset(dataset_id: str, dm: DatasetManager = Depends(get_dataset_manager)):
    return dm.get_dataset_info(dataset_id)


@router.post(
    "/{dataset_id}/default_version",
    response_model=bool,
    summary="Изменение стандартной версии датасета",
    response_description="True, если изменения успешно внесены",
)
def new_default_version(
    dataset_id: str,
    default_version: str,
    dm: DatasetManager = Depends(get_dataset_manager)
):
    ds = dm.get_dataset_info(dataset_id)
    ds.default_version_id = default_version
    dm.change_dataset_info(ds)
    
    return True

@router.delete(
    "/{dataset_id}", 
    response_model=bool,
    summary="Удалить датасет",
    description="Удаляет указанный датасет из системы",
    response_description="True, если датасет был успешно удалён",
)
def delete_dataset(dataset_id: str, dm: DatasetManager = Depends(get_dataset_manager)):
    return dm.drop_dataset(dataset_id)

@router.post(
    "/new", 
    summary="Создать датасет",
    description="Создаёт новый датасет из загружает файл данных.",
    response_description="True, если датасет успешно создан",
)
def create_dataset(
    new_dataset: NewDataset, 
    dm: DatasetManager = Depends(get_dataset_manager),
):
    try:
        dm.add_new_dataset(new_dataset)
        return True
    except CoreException as e:
        logger.error(f"\nОшибка: {e.message}\nДетали: {e.detail}")
        return HTTPException(
            status_code=e.status_code, 
            detail=e.message
        )

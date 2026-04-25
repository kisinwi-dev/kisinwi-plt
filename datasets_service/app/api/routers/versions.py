from fastapi import APIRouter, Depends
from typing import List

from app.logs import get_logger
from app.core.services import DatasetManager
from app.api.deps import get_dataset_manager
from app.api.schemas.dataset import Version
from app.api.schemas.dataset_new import NewVersion

logger = get_logger(__name__)
router = APIRouter(prefix="/datasets/{dataset_id}/versions", tags=["Version"])

@router.get(
    "/", 
    response_model=List[Version],
    summary="Получение списка метаданных версий",
    description="Возвращает список метаданных всех доступных версий датасета",
    response_description="Список метаданных версий",
)
def list_versions(
        dataset_id: str,
        dm: DatasetManager = Depends(get_dataset_manager)
):
    return dm.get_dataset_info(dataset_id).versions

@router.get(
    "/{version_id}", 
    response_model=Version,
    summary="Получить метаданные версии",
    description="Возвращает метаданные указанной версии по её идентификатору",
    response_description="Метаданные версии",
)
def get_infp_version(
        dataset_id: str,
        version_id: str,
        dm: DatasetManager = Depends(get_dataset_manager)
):
    datasets = dm.get_dataset_info(dataset_id)
    for version in datasets.versions:
        if version.version_id == version_id:
            return version

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
    summary="Создать новую версию данных",
    description="""
Добавляет новую версию данных к существующему датасету. Данные должны быть предварительно загружены через эндпоинт `/upload`.

**Пример использования:**

1. Загружаем данные через `/upload` с параметром `id_data = 'v2.0'`
2. Создаём новую версию через `/datasets/my_dataset/versions/new` с телом:
```json
{
  "version_id": "v2.0",
  "description": "Версия с очищенными данными"
}
""",
    response_description="True, если версия успешно создана",
)
def create_version(
    dataset_id: str,
    new_dataset: NewVersion, 
    dm: DatasetManager = Depends(get_dataset_manager),
):
    return dm.add_new_version(dataset_id, new_dataset)

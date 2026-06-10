from fastapi import APIRouter, Depends
from typing import List

from app.logs import get_logger
from app.core.services import DatasetManager
from app.api.deps import get_dataset_manager
from app.api.schemas.dataset import VersionResponse, SplitSummaryResponse
from app.api.schemas.splits import (
    SplitCountsResponse, SplitBalanceResponse,
    ClassDistributionResponse, ImageSizeStatsResponse
)
from app.api.schemas.dataset_new import NewVersion

logger = get_logger(__name__)
router = APIRouter(prefix="/datasets/{dataset_id}/versions", tags=["Version"])

@router.get(
    "/", 
    response_model=List[VersionResponse],
    summary="Получение списка метаданных версий",
    description="Возвращает список метаданных всех доступных версий датасета",
    response_description="Список метаданных версий",
)
def list_versions(
        dataset_id: str,
        dm: DatasetManager = Depends(get_dataset_manager)
):
    return dm.get_dataset_response_info(dataset_id).versions

@router.get(
    "/{version_id}", 
    response_model=VersionResponse,
    summary="Получить метаданные версии",
    description="Возвращает метаданные указанной версии по её идентификатору",
    response_description="Метаданные версии",
)
def get_version_all_metadata(
        dataset_id: str,
        version_id: str,
        dm: DatasetManager = Depends(get_dataset_manager)
):
    return dm.get_version_info(dataset_id, version_id)

@router.get(
    "/{version_id}/splits", 
    summary="Получить статистику по сплитам",
    description="""

    Возвращает детальную информацию о разбиении датасета на сплиты:

    - Общее количество изображений

    - Баланс классов в каждом сплите

    - Распределение по классам
    
    - Статистику размеров изображений
    """,
    response_description="Полная информация о разбиении датасета",
    response_model=SplitSummaryResponse
)
def get_version_splits(
        dataset_id: str,
        version_id: str,
        dm: DatasetManager = Depends(get_dataset_manager)
):
    return dm.get_version_split_summary(dataset_id, version_id)

@router.get(
    "/{version_id}/splits/count",
    summary="Получить количество изображений по сплитам",
    description="Возвращает общее количество изображений версии и количество в каждом сплите",
    response_description="Количество изображений по сплитам",
    response_model=SplitCountsResponse
)
def get_version_split_counts(
        dataset_id: str,
        version_id: str,
        dm: DatasetManager = Depends(get_dataset_manager)
):
    return dm.get_version_split_counts(dataset_id, version_id)

@router.get(
    "/{version_id}/splits/balance",
    summary="Получить баланс классов по сплитам",
    description="Возвращает коэффициент баланса классов в каждом сплите и общий баланс версии",
    response_description="Баланс классов по сплитам",
    response_model=SplitBalanceResponse
)
def get_version_split_balance(
        dataset_id: str,
        version_id: str,
        dm: DatasetManager = Depends(get_dataset_manager)
):
    return dm.get_version_split_balance(dataset_id, version_id)

@router.get(
    "/{version_id}/splits/distribution",
    summary="Получить распределение классов",
    description="Возвращает распределение по классам в каждом сплите",
    response_description="Распределение классов по сплитам",
    response_model=ClassDistributionResponse
)
def get_version_class_distribution(
        dataset_id: str,
        version_id: str,
        dm: DatasetManager = Depends(get_dataset_manager)
):
    return dm.get_version_class_distribution(dataset_id, version_id)

@router.get(
    "/{version_id}/splits/size-stats",
    summary="Получить статистику размеров изображений",
    description="Возвращает статистику размеров изображений в каждом сплите",
    response_description="Статистика размеров изображений по сплитам",
    response_model=ImageSizeStatsResponse
)
def get_version_image_size_stats(
        dataset_id: str,
        version_id: str,
        dm: DatasetManager = Depends(get_dataset_manager)
):
    return dm.get_version_image_size_stats(dataset_id, version_id)

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

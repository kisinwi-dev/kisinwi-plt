from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List

from app.logs import get_logger
from app.core.services import DatasetManager
from app.core.exception.base import CoreException
from app.api.deps import get_dataset_manager
from app.api.schemas.dataset import VersionResponse, SplitSummaryResponse
from app.api.schemas.splits import (
    SplitCountsResponse, SplitBalanceResponse,
    ClassDistributionResponse, ImageSizeStatsResponse
)
from app.api.schemas.comparison import (
    VersionComparisonResponse, CountsComparisonResponse,
    DistributionComparisonResponse, BalanceComparisonResponse,
    SizeStatsComparisonResponse, FilesDiffResponse
)
from app.api.schemas.dataset_new import NewVersion

logger = get_logger(__name__)
router = APIRouter(prefix="/datasets/{dataset_id}/versions", tags=["Version"])

def _to_http_error(e: CoreException) -> HTTPException:
    logger.error(f"\nОшибка: {e.message}\nДетали: {e.detail}")
    return HTTPException(
        status_code=e.status_code,
        detail=f"{e.message} {e.detail}" if e.detail else e.message
    )

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
    "/compare",
    response_model=VersionComparisonResponse,
    summary="Сравнить две версии датасета",
    description="""
Возвращает полную сводку сравнения двух версий одного датасета:

- Изменение количества изображений по сплитам и классам

- Изменения состава классов и drift-метрики распределений (JS divergence, PSI)

- Изменение баланса классов

- Изменение размеров и форматов изображений

- Счётчики по-файлового diff (полные списки — в `/compare/files`)
    """,
    response_description="Полная сводка сравнения версий",
)
def compare_versions(
        dataset_id: str,
        from_version: str = Query(..., alias="from", description="ID базовой версии"),
        to_version: str = Query(..., alias="to", description="ID сравниваемой версии"),
        dm: DatasetManager = Depends(get_dataset_manager)
):
    try:
        return dm.compare_versions(dataset_id, from_version, to_version)
    except CoreException as e:
        raise _to_http_error(e)

@router.get(
    "/compare/counts",
    response_model=CountsComparisonResponse,
    summary="Сравнить количество изображений двух версий",
    description="Возвращает изменение количества изображений между версиями: общее, по сплитам и по классам",
    response_description="Сравнение количества изображений",
)
def compare_version_counts(
        dataset_id: str,
        from_version: str = Query(..., alias="from", description="ID базовой версии"),
        to_version: str = Query(..., alias="to", description="ID сравниваемой версии"),
        dm: DatasetManager = Depends(get_dataset_manager)
):
    try:
        return dm.compare_version_counts(dataset_id, from_version, to_version)
    except CoreException as e:
        raise _to_http_error(e)

@router.get(
    "/compare/distribution",
    response_model=DistributionComparisonResponse,
    summary="Сравнить распределения классов двух версий",
    description="Возвращает изменения состава классов и drift-метрики распределений (JS divergence, PSI) по сплитам",
    response_description="Сравнение распределений классов",
)
def compare_version_distribution(
        dataset_id: str,
        from_version: str = Query(..., alias="from", description="ID базовой версии"),
        to_version: str = Query(..., alias="to", description="ID сравниваемой версии"),
        dm: DatasetManager = Depends(get_dataset_manager)
):
    try:
        return dm.compare_version_distribution(dataset_id, from_version, to_version)
    except CoreException as e:
        raise _to_http_error(e)

@router.get(
    "/compare/balance",
    response_model=BalanceComparisonResponse,
    summary="Сравнить баланс классов двух версий",
    description="Возвращает изменение коэффициента баланса классов: общего и по сплитам",
    response_description="Сравнение баланса классов",
)
def compare_version_balance(
        dataset_id: str,
        from_version: str = Query(..., alias="from", description="ID базовой версии"),
        to_version: str = Query(..., alias="to", description="ID сравниваемой версии"),
        dm: DatasetManager = Depends(get_dataset_manager)
):
    try:
        return dm.compare_version_balance(dataset_id, from_version, to_version)
    except CoreException as e:
        raise _to_http_error(e)

@router.get(
    "/compare/size-stats",
    response_model=SizeStatsComparisonResponse,
    summary="Сравнить размеры и форматы изображений двух версий",
    description="Возвращает изменение количества изображений по форматам и по размерам (WxH) в каждом сплите",
    response_description="Сравнение размеров и форматов изображений",
)
def compare_version_size_stats(
        dataset_id: str,
        from_version: str = Query(..., alias="from", description="ID базовой версии"),
        to_version: str = Query(..., alias="to", description="ID сравниваемой версии"),
        dm: DatasetManager = Depends(get_dataset_manager)
):
    try:
        return dm.compare_version_size_stats(dataset_id, from_version, to_version)
    except CoreException as e:
        raise _to_http_error(e)

@router.get(
    "/compare/files",
    response_model=FilesDiffResponse,
    summary="Сравнить файлы двух версий",
    description="Возвращает по-файловый diff между версиями по относительным путям (split/class/filename): добавленные и удалённые файлы",
    response_description="По-файловый diff версий",
)
def compare_version_files(
        dataset_id: str,
        from_version: str = Query(..., alias="from", description="ID базовой версии"),
        to_version: str = Query(..., alias="to", description="ID сравниваемой версии"),
        dm: DatasetManager = Depends(get_dataset_manager)
):
    try:
        return dm.compare_version_files(dataset_id, from_version, to_version)
    except CoreException as e:
        raise _to_http_error(e)

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
    try:
        return dm.add_new_version(dataset_id, new_dataset)
    except CoreException as e:
        dm.drop_cache()
        raise _to_http_error(e)

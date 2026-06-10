from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional

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
from app.api.schemas.dataset_new import NewVersion, VersionUpdate
from app.api.schemas.integrity import IntegrityReportResponse
from app.api.schemas.files import VersionFilesResponse
from app.api.schemas.splits import SplitType

logger = get_logger(__name__)
router = APIRouter(prefix="/datasets/{dataset_id}/versions")

def _to_http_error(e: CoreException) -> HTTPException:
    logger.error(f"\nОшибка: {e.message}\nДетали: {e.detail}")
    return HTTPException(
        status_code=e.status_code,
        detail=f"{e.message} {e.detail}" if e.detail else e.message
    )

@router.get(
    "/",
    tags=["Versions"],
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
    tags=["Version Compare"],
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
    tags=["Version Compare"],
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
    tags=["Version Compare"],
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
    tags=["Version Compare"],
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
    tags=["Version Compare"],
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
    tags=["Version Compare"],
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
    tags=["Versions"],
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
    tags=["Version Stats"],
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
    tags=["Version Stats"],
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
    tags=["Version Stats"],
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
    tags=["Version Stats"],
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
    tags=["Version Stats"],
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

@router.get(
    "/{version_id}/integrity",
    tags=["Version Stats"],
    summary="Получить отчёт целостности данных",
    description="""
Возвращает детальный отчёт о целостности данных версии (по SHA256-хешам файлов):

- Группы одинаковых файлов внутри одного сплита (дубликаты)

- Изображения, встречающиеся в нескольких сплитах (train/test leakage)
    """,
    response_description="Отчёт о дубликатах и утечках между сплитами",
    response_model=IntegrityReportResponse
)
def get_version_integrity(
        dataset_id: str,
        version_id: str,
        dm: DatasetManager = Depends(get_dataset_manager)
):
    try:
        return dm.get_version_integrity(dataset_id, version_id)
    except CoreException as e:
        raise _to_http_error(e)

@router.get(
    "/{version_id}/files",
    tags=["Version Stats"],
    summary="Получить список файлов версии",
    description="""
Возвращает страницу списка файлов версии (относительные пути `split/class/filename`).

- `split` — фильтр по одному сплиту (train/val/test)
- `limit`/`offset` — пагинация, `total` в ответе считается с учётом фильтра
""",
    response_description="Страница списка файлов версии",
    response_model=VersionFilesResponse
)
def get_version_files(
        dataset_id: str,
        version_id: str,
        split: Optional[SplitType] = Query(None, description="Фильтр по сплиту"),
        limit: int = Query(100, ge=1, le=10000, description="Размер страницы"),
        offset: int = Query(0, ge=0, description="Смещение от начала списка"),
        dm: DatasetManager = Depends(get_dataset_manager)
):
    try:
        return dm.get_version_files(dataset_id, version_id, split, limit, offset)
    except CoreException as e:
        raise _to_http_error(e)

@router.patch(
    "/{version_id}",
    tags=["Versions"],
    response_model=VersionResponse,
    summary="Изменить название/описание версии",
    description="Обновляет name и/или description версии. ID и данные версии не меняются.",
    response_description="Обновлённые метаданные версии",
)
def update_version(
        dataset_id: str,
        version_id: str,
        update: VersionUpdate,
        dm: DatasetManager = Depends(get_dataset_manager)
):
    try:
        return dm.update_version_info(dataset_id, version_id, update.name, update.description)
    except CoreException as e:
        raise _to_http_error(e)

@router.delete(
    "/{version_id}",
    tags=["Versions"],
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
    tags=["Versions"],
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
    new_version: NewVersion,
    dm: DatasetManager = Depends(get_dataset_manager),
):
    try:
        return dm.add_new_version(dataset_id, new_version)
    except CoreException as e:
        # данные чистим только при ошибке валидации: при сбое сохранения
        # метаданных они уже возвращены в temp и пригодны для повтора
        if e.status_code == 400:
            dm.drop_cache(new_version.id_data)
        raise _to_http_error(e)

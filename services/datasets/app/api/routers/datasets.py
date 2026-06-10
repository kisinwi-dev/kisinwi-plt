from fastapi import APIRouter, Depends, HTTPException, Form, Query, Response
from typing import List, Optional

from app.logs import get_logger
from app.core.services import DatasetManager
from app.core.exception.base import CoreException
from app.api.deps import get_dataset_manager
from app.api.schemas.dataset import DatasetResponse
from app.api.schemas.dataset_new import NewDataset, DatasetUpdate

logger = get_logger(__name__)
router = APIRouter(prefix="/datasets", tags=["Datasets"])

@router.get(
    "/",
    response_model=List[DatasetResponse],
    summary="Получение списка метаданных датасетов",
    description="""
Возвращает страницу списка датасетов.

- `limit`/`offset` — пагинация
- `search` — фильтр по подстроке в id или name (без учёта регистра)
- Общее количество (с учётом фильтра) — в заголовке ответа `X-Total-Count`
""",
    response_description="Список метаданных датасетов",
)
def list_datasets(
    response: Response,
    limit: int = Query(50, ge=1, le=500, description="Размер страницы"),
    offset: int = Query(0, ge=0, description="Смещение от начала списка"),
    search: Optional[str] = Query(None, description="Подстрока для поиска по id/name"),
    dm: DatasetManager = Depends(get_dataset_manager)
):
    datasets, total = dm.list_datasets_response(limit=limit, offset=offset, search=search)
    response.headers["X-Total-Count"] = str(total)
    return datasets

@router.get(
    "/{dataset_id}", 
    response_model=DatasetResponse,
    summary="Получить метаданные датасета",
    description="Возвращает метаданные указанного датасета по его идентификатору",
    response_description="Метаданные датасета",
)
def get_dataset(
    dataset_id: str,
    dm: DatasetManager = Depends(get_dataset_manager)
):
    return dm.get_dataset_response_info(dataset_id)

@router.post(
    "/{dataset_id}/default_version",
    response_model=bool,
    summary="Изменение стандартной версии датасета",
    description="Устанавливает указанную версию как стандартную для датасета",
    response_description="True, если изменения успешно внесены",
)
def new_default_version(
    dataset_id: str,
    default_version: str,
    dm: DatasetManager = Depends(get_dataset_manager)
):
    ds = dm._get_dataset_info(dataset_id)
    ds.default_version_id = default_version
    dm.change_dataset_info(ds)
    return True

@router.patch(
    "/{dataset_id}",
    response_model=DatasetResponse,
    summary="Изменить название/описание датасета",
    description="Обновляет name и/или description датасета. ID и расположение на диске не меняются.",
    response_description="Обновлённые метаданные датасета",
)
def update_dataset(
    dataset_id: str,
    update: DatasetUpdate,
    dm: DatasetManager = Depends(get_dataset_manager)
):
    try:
        return dm.update_dataset_info(dataset_id, update.name, update.description)
    except CoreException as e:
        raise HTTPException(
            status_code=e.status_code,
            detail=f"{e.message} {e.detail}" if e.detail else e.message
        )

@router.delete(
    "/{dataset_id}",
    response_model=bool,
    summary="Удалить датасет",
    description="Удаляет указанный датасет из системы",
    response_description="True, если датасет был успешно удалён",
)
def delete_dataset(
    dataset_id: str, 
    dm: DatasetManager = Depends(get_dataset_manager)
):
    return dm.drop_dataset(dataset_id)

@router.post(
    "/new", 
    summary="Создать датасет",
    description="""
Создаёт новый датасет из ранее загруженных данных из /upload.

**Пример использования:**

1. Загружаем данные через `/upload` с параметром `id_data = 'my_dataset'`
2. Создаём датасет через `/datasets/new` с телом:
```json
{
  "id": "my_dataset",
  "title": "Мой датасет",
  "description": "Тестовые данные"
  ...
}
```
""",
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
        # данные чистим только при ошибке валидации: при сбое сохранения
        # метаданных они уже возвращены в temp и пригодны для повтора
        if e.status_code == 400:
            dm.drop_cache(new_dataset.version.id_data)
        raise HTTPException(
            status_code=e.status_code,
            detail=f"{e.message} {e.detail}" if e.detail else e.message
        )

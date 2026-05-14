from crewai.tools import tool

from .utils import (
    get_json, handle_errors, health, 
    get_sample_sizes_for_all_data
)
from app.logs import get_logger

logger = get_logger(__name__)

@tool("GetDatasetInfo")
@handle_errors
def get_dataset_info(dataset_id: str) -> dict:
    """
    НАЗНАЧЕНИЕ: Получить полную информацию о датасете по его ID.

    КОГДА ИСПОЛЬЗОВАТЬ:
    - Когда нужно узнать детали конкретного датасета
    - Перед обучением модели, чтобы понять структуру данных
    - Для получения списка классов, размера датасета, описания
    - Когда нужно проверить существование датасета

    ВХОДНЫЕ ДАННЫЕ:
    - dataset_id (str): Уникальный идентификатор датасета.
      Пример: "human_face_emotions"

    ВОЗВРАЩАЕТ:
    - dict с информацией о датасете

    ПРИМЕР ВЫЗОВА:
    get_dataset_info("human_face_emotions")

    ВАЖНЫЕ ЗАМЕЧАНИЯ:
    - Используй этот инструмент перед работой с конкретными версиями
    """
    logger.info(f"🔧 GetDatasetInfo вызван для датасета: {dataset_id}")
    return get_json(f"/api/datasets/{dataset_id}")

    
@tool("GetDatasetVersionInfo")
@handle_errors
def get_version_info(
    dataset_id: str,
    version_id: str
) -> dict:
    """
    НАЗНАЧЕНИЕ: Получить информацию о конкретной версии датасета.

    КОГДА ИСПОЛЬЗОВАТЬ:
    - Когда нужно узнать детали версии датасета
    - Для получения размера версии, количества сэмплов
    - Перед обучением модели на конкретной версии
    - Для проверки, какие классы входят в версию
    
    ВХОДНЫЕ ДАННЫЕ:
    - dataset_id (str): ID датасета.
      Пример: "human_face_emotions"
    - version_id (str): ID версии датасета.
      Пример: "v_1"
    
    ВОЗВРАЩАЕТ:
    - dict с информацией о версии
    
    ПРИМЕР ВЫЗОВА:
    get_version_info("human_face_emotions", "v_1")
    
    ВАЖНЫЕ ЗАМЕЧАНИЯ:
    - Всегда проверяй существование версии перед вызовом
    - class_distribution важен для выявления дисбаланса классов
    """
    logger.info(f"🔧 GetDatasetVersionInfo вызван для датасета '{dataset_id}' и версии '{version_id}'")
    return get_json(f"/api/datasets/{dataset_id}/versions/{version_id}")

    
@tool("GetDatasetVersionSplitInfo")
@handle_errors
def get_version_split_info(
    dataset_id: str,
    version_id: str
) -> dict:
    """
    НАЗНАЧЕНИЕ: Получить информацию о разбиении датасета на train/val/test.
    
    КОГДА ИСПОЛЬЗОВАТЬ:
    - Перед обучением модели, чтобы понять размеры выборок
    - Для проверки сбалансированности разбиения
    - Когда нужно узнать, сколько сэмплов в train/val/test
    - Для планирования стратегии обучения
    
    ВХОДНЫЕ ДАННЫЕ:
    - dataset_id (str): ID датасета.
      Пример: "human_face_emotions"
    - version_id (str): ID версии датасета.
      Пример: "v_1"
    
    ВОЗВРАЩАЕТ:
    - dict с информацией о сэмплах
    
    ПРИМЕР ВЫЗОВА:
    get_version_split_info("human_face_emotions", "v_1")
    
    ВАЖНЫЕ ЗАМЕЧАНИЯ:
    - Используй для проверки, что разбиение сбалансированное
    - Для задач классификации важно, чтобы распределение классов было равномерным
    """
    logger.info(f"🔧 GetDatasetVersionSplitInfo вызван для датасета '{dataset_id}' и версии '{version_id}'")
    json = get_json(f"/api/datasets/{dataset_id}/versions/{version_id}")
    return get_sample_sizes_for_all_data(json)


@tool("ListDatasets")
@handle_errors
def list_datasets() -> dict:
    """
    НАЗНАЧЕНИЕ: Получить список всех доступных датасетов в системе.
    
    КОГДА ИСПОЛЬЗОВАТЬ:
    - Когда нужно узнать, какие датасеты доступны
    - Для выбора датасета перед началом работы
    - Для поиска конкретного датасета по названию
    - Для обзора всех доступных данных
    
    ВХОДНЫЕ ДАННЫЕ:
    - Нет входных параметров
    
    ВОЗВРАЩАЕТ:
    - dict с иноформацией о имеющихся датасетах
    
    ПРИМЕР ВЫЗОВА:
    list_datasets()
    
    ВАЖНЫЕ ЗАМЕЧАНИЯ:
    - Сначала вызови этот инструмент, чтобы узнать доступные датасеты
    - Затем используй GetDatasetInfo для конкретного датасета
    """
    logger.info(f"🔧 ListDatasets вызван")
    return get_json("/api/datasets/")

def get_dataset_info_classes(dataset_id: str) -> list:
    """
    Получить всю информацию о датасете по ID
    
    Args:
        dataset_id: Id датасета
        
    Returns:
        Dict с информацией о датасете или ошибкой
    """    
    return get_json(f"/api/datasets/{dataset_id}")["class_names"]
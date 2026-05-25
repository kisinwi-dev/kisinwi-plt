from crewai.tools import BaseTool
from typing import Dict, Any

from ..utils import (
    get_json, handle_errors
)
from app.config import config_url
from app.logs import get_logger

logger = get_logger(__name__)

DATASETS_URL = config_url.DATASETS['url']

class GetDatasetDetailsTool(BaseTool):
    """Инструмент для получения полной информации о датасете по его ID"""

    name: str = "GetDatasetDetails"
    description: str = """
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

    ВАЖНЫЕ ЗАМЕЧАНИЯ:
    - Используй этот инструмент перед работой с конкретными версиями
    """

    @handle_errors(DATASETS_URL)
    def _run(self, dataset_id: str) -> Dict[str, Any]:
        return get_json(f"{DATASETS_URL}/api/datasets/{dataset_id}")

    async def _arun(self, dataset_id: str) -> Dict[str, Any]:
        return get_json(f"{DATASETS_URL}/api/datasets/{dataset_id}")


class GetDatasetVersionDetailsTool(BaseTool):
    """Инструмент для получения информации о конкретной версии датасета"""

    name: str = "GetDatasetVersionDetails"
    description: str = """
    НАЗНАЧЕНИЕ: Получить информацию о конкретной версии датасета.

    КОГДА ИСПОЛЬЗОВАТЬ:
    - Когда нужно узнать детали версии датасета
    - Для получения размера версии, количества сэмплов
    - Перед обучением модели на конкретной версии

    ВХОДНЫЕ ДАННЫЕ:
    - dataset_id (str): ID датасета.
      Пример: "human_face_emotions"
    - version_id (str): ID версии датасета.
      Пример: "v_1"

    ВОЗВРАЩАЕТ:
    - dict с информацией о версии

    ВАЖНЫЕ ЗАМЕЧАНИЯ:
    - Всегда проверяй существование версии перед вызовом
    """

    @handle_errors(DATASETS_URL)
    def _run(self, dataset_id: str, version_id: str) -> Dict[str, Any]:
        return get_json(f"{DATASETS_URL}/api/datasets/{dataset_id}/versions/{version_id}")

    async def _arun(self, dataset_id: str, version_id: str) -> Dict[str, Any]:
        return get_json(f"{DATASETS_URL}/api/datasets/{dataset_id}/versions/{version_id}")


class GetDatasetSplitSizesTool(BaseTool):
    """Инструмент для получения информации о разбиении версии датасета на train/val/test"""

    name: str = "GetDatasetSplitSizes"
    description: str = """
    НАЗНАЧЕНИЕ: Получить информацию о разбиении версии датасета на train/val/test.

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

    ВАЖНЫЕ ЗАМЕЧАНИЯ:
    - Используй для проверки, что разбиение сбалансированное
    - Для задач классификации важно, чтобы распределение классов было равномерным
    """

    @handle_errors(DATASETS_URL)
    def _run(self, dataset_id: str, version_id: str) -> Dict[str, Any]:
        json_data = get_json(f"{DATASETS_URL}/api/datasets/{dataset_id}/versions/{version_id}")
        return get_sample_sizes_for_all_data(json_data)

    async def _arun(self, dataset_id: str, version_id: str) -> Dict[str, Any]:
        json_data = get_json(f"{DATASETS_URL}/api/datasets/{dataset_id}/versions/{version_id}")
        return get_sample_sizes_for_all_data(json_data)


class ListAllDatasetsTool(BaseTool):
    """Инструмент для получения всех доступных датасетов"""

    name: str = "ListAllDatasets"
    description: str = """
    НАЗНАЧЕНИЕ: Получить всех доступных датасетов в системе.

    КОГДА ИСПОЛЬЗОВАТЬ:
    - Когда нужно узнать, какие датасеты доступны
    - Для выбора датасета перед началом работы
    - Для поиска конкретного датасета по названию
    - Для обзора всех доступных данных

    ВХОДНЫЕ ДАННЫЕ:
    - Нет входных параметров

    ВОЗВРАЩАЕТ:
    - dict с информацией об имеющихся датасетах
    """

    @handle_errors(DATASETS_URL)
    def _run(self) -> Dict[str, Any]:
        return get_json(f"{DATASETS_URL}/api/datasets/")

    async def _arun(self) -> Dict[str, Any]:
        return get_json(f"{DATASETS_URL}/api/datasets/")

# __WARNING__ ТРЕБУЕТСЯ РЕАЛИЗОВАТЬ ТАКОЙ МЕТОД В СЕРВИСЕ С ДАТАСЕТАМИ
def get_sample_sizes_for_all_data(dataset_info: dict) -> dict:
    """
    Извлекает информацию о размерах выборки из полной информации о датасете.
    
    Args:
        dataset_info: Полный JSON с информацией о датасете
        
    Returns:
        Словарь с информацией о размерах выборки
    """
    
    result = {
        "version_id": dataset_info["version_id"],
        "total_samples": dataset_info["num_samples"],
        "splits": {}
    }
    
    # Извлекаем информацию по каждому сплиту
    for split_name, split_data in dataset_info["splits"].items():
        class_distribution = split_data["class_distribution"]
        
        # Общее количество в сплите
        total_in_split = sum(cls.get("count", 0) for cls in class_distribution)
        
        # Распределение по классам
        class_counts = {
            cls["class_name"]: cls["count"]
            for cls in class_distribution
        }
        
        result["splits"][split_name] = {
            "total": total_in_split,
            "classes": class_counts
        }
    
    return result
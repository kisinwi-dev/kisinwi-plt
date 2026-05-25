from crewai.tools import BaseTool
from pydantic import Field
from typing import Dict, Any

from .utils import (
    get_json, handle_errors, 
    get_sample_sizes_for_all_data
)
from app.logs import get_logger

logger = get_logger(__name__)

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

    @handle_errors
    def _run(self, dataset_id: str) -> Dict[str, Any]:
        """Выполнение инструмента"""
        return get_json(f"/api/datasets/{dataset_id}")

    async def _arun(self, dataset_id: str) -> Dict[str, Any]:
        """Асинхронная версия"""
        return get_json(f"/api/datasets/{dataset_id}")


class GetDatasetVersionDetailsTool(BaseTool):
    """Инструмент для получения информации о конкретной версии датасета"""

    name: str = "GetDatasetVersionDetails"
    description: str = """
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

    ВАЖНЫЕ ЗАМЕЧАНИЯ:
    - Всегда проверяй существование версии перед вызовом
    - class_distribution важен для выявления дисбаланса классов
    """

    @handle_errors
    def _run(self, dataset_id: str, version_id: str) -> Dict[str, Any]:
        """Выполнение инструмента"""
        return get_json(f"/api/datasets/{dataset_id}/versions/{version_id}")

    async def _arun(self, dataset_id: str, version_id: str) -> Dict[str, Any]:
        """Асинхронная версия"""
        return get_json(f"/api/datasets/{dataset_id}/versions/{version_id}")


class GetDatasetSplitSizesTool(BaseTool):
    """Инструмент для получения информации о разбиении датасета на train/val/test"""

    name: str = "GetDatasetSplitSizes"
    
    description: str = """
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

    ВАЖНЫЕ ЗАМЕЧАНИЯ:
    - Используй для проверки, что разбиение сбалансированное
    - Для задач классификации важно, чтобы распределение классов было равномерным
    """

    @handle_errors
    def _run(self, dataset_id: str, version_id: str) -> Dict[str, Any]:
        """Выполнение инструмента"""
        json_data = get_json(f"/api/datasets/{dataset_id}/versions/{version_id}")
        return get_sample_sizes_for_all_data(json_data)

    async def _arun(self, dataset_id: str, version_id: str) -> Dict[str, Any]:
        """Асинхронная версия"""
        json_data = get_json(f"/api/datasets/{dataset_id}/versions/{version_id}")
        return get_sample_sizes_for_all_data(json_data)


class ListAllDatasetsTool(BaseTool):
    """Инструмент для получения списка всех доступных датасетов"""

    name: str = "ListAllDatasets"

    description: str = """
    НАЗНАЧЕНИЕ: Получить список всех доступных датасетов в системе.

    КОГДА ИСПОЛЬЗОВАТЬ:
    - Когда нужно узнать, какие датасеты доступны
    - Для выбора датасета перед началом работы
    - Для поиска конкретного датасета по названию
    - Для обзора всех доступных данных

    ВХОДНЫЕ ДАННЫЕ:
    - Нет входных параметров

    ВОЗВРАЩАЕТ:
    - dict с информацией о имеющихся датасетах

    ВАЖНЫЕ ЗАМЕЧАНИЯ:
    - Сначала вызови этот инструмент, чтобы узнать доступные датасеты
    - Затем используй GetDatasetDetailsTool для конкретного датасета
    """

    @handle_errors
    def _run(self) -> Dict[str, Any]:
        """Выполнение инструмента"""
        return get_json("/api/datasets/")

    async def _arun(self) -> Dict[str, Any]:
        """Асинхронная версия"""
        return get_json("/api/datasets/")
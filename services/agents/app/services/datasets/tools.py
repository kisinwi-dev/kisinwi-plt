import asyncio
from crewai.tools import BaseTool
from typing import Dict, Any

from ..utils import (
    get_json, tool_response
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

    @tool_response(DATASETS_URL)
    def _run(self, dataset_id: str) -> str:
        url = f"{DATASETS_URL}/datasets/{dataset_id}"
        return get_json(url) # type: ignore[return-value]  Декоратор преобразет ответ в str

    async def _arun(self, dataset_id: str) -> str:
        return await asyncio.to_thread(self._run, dataset_id)


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

    @tool_response(DATASETS_URL)
    def _run(self, dataset_id: str, version_id: str) -> str:
        url = f"{DATASETS_URL}/datasets/{dataset_id}/versions/{version_id}"
        return get_json(url) # type: ignore[return-value]  Декоратор преобразет ответ в str

    async def _arun(self, dataset_id: str, version_id: str) -> str:
        return await asyncio.to_thread(self._run, dataset_id, version_id)


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

    @tool_response(DATASETS_URL)
    def _run(self, dataset_id: str, version_id: str) -> str:
        url = f"{DATASETS_URL}/datasets/{dataset_id}/versions/{version_id}/splits"
        return get_json(url) # type: ignore[return-value]  Декоратор преобразет ответ в str

    async def _arun(self, dataset_id: str, version_id: str) -> str:
        return await asyncio.to_thread(self._run, dataset_id, version_id)


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

    @tool_response(DATASETS_URL)
    def _run(self) -> str:
        url = f"{DATASETS_URL}/datasets/"
        return get_json(url) # type: ignore[return-value]  Декоратор преобразет ответ в str

    async def _arun(self) -> str:
        return await asyncio.to_thread(self._run)

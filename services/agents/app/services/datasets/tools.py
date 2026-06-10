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


class GetDatasetSplitCountsTool(BaseTool):
    """Инструмент для получения количества изображений по сплитам версии датасета"""

    name: str = "GetDatasetSplitCounts"
    description: str = """
    НАЗНАЧЕНИЕ: Получить общее количество изображений версии и количество в каждом сплите (train/val/test).

    КОГДА ИСПОЛЬЗОВАТЬ:
    - Когда нужны только размеры выборок, без остальной статистики
    - Для оценки достаточности данных перед обучением
    - Для проверки пропорций разбиения train/val/test

    ВХОДНЫЕ ДАННЫЕ:
    - dataset_id (str): ID датасета.
      Пример: "human_face_emotions"
    - version_id (str): ID версии датасета.
      Пример: "v_1"

    ВОЗВРАЩАЕТ:
    - dict с общим количеством изображений и количеством по сплитам

    ВАЖНЫЕ ЗАМЕЧАНИЯ:
    - Это более лёгкий запрос, чем GetDatasetSplitSizes — используй его, если нужны только размеры
    """

    @tool_response(DATASETS_URL)
    def _run(self, dataset_id: str, version_id: str) -> str:
        url = f"{DATASETS_URL}/datasets/{dataset_id}/versions/{version_id}/splits/count"
        return get_json(url) # type: ignore[return-value]  Декоратор преобразет ответ в str

    async def _arun(self, dataset_id: str, version_id: str) -> str:
        return await asyncio.to_thread(self._run, dataset_id, version_id)


class GetDatasetSplitBalanceTool(BaseTool):
    """Инструмент для получения баланса классов по сплитам версии датасета"""

    name: str = "GetDatasetSplitBalance"
    description: str = """
    НАЗНАЧЕНИЕ: Получить коэффициент баланса классов в каждом сплите и общий баланс версии.

    КОГДА ИСПОЛЬЗОВАТЬ:
    - Для проверки сбалансированности классов перед обучением
    - Когда нужно решить, требуются ли веса классов или аугментация
    - Для выявления перекоса классов в train/val/test

    ВХОДНЫЕ ДАННЫЕ:
    - dataset_id (str): ID датасета.
      Пример: "human_face_emotions"
    - version_id (str): ID версии датасета.
      Пример: "v_1"

    ВОЗВРАЩАЕТ:
    - dict с коэффициентами баланса классов по сплитам и общим балансом

    ВАЖНЫЕ ЗАМЕЧАНИЯ:
    - Для задач классификации важно, чтобы распределение классов было равномерным
    """

    @tool_response(DATASETS_URL)
    def _run(self, dataset_id: str, version_id: str) -> str:
        url = f"{DATASETS_URL}/datasets/{dataset_id}/versions/{version_id}/splits/balance"
        return get_json(url) # type: ignore[return-value]  Декоратор преобразет ответ в str

    async def _arun(self, dataset_id: str, version_id: str) -> str:
        return await asyncio.to_thread(self._run, dataset_id, version_id)


class GetDatasetClassDistributionTool(BaseTool):
    """Инструмент для получения распределения классов по сплитам версии датасета"""

    name: str = "GetDatasetClassDistribution"
    description: str = """
    НАЗНАЧЕНИЕ: Получить распределение изображений по классам в каждом сплите.

    КОГДА ИСПОЛЬЗОВАТЬ:
    - Когда нужно точное количество изображений каждого класса в train/val/test
    - Для анализа редких классов и принятия решения об аугментации
    - Для проверки, что все классы представлены в каждом сплите

    ВХОДНЫЕ ДАННЫЕ:
    - dataset_id (str): ID датасета.
      Пример: "human_face_emotions"
    - version_id (str): ID версии датасета.
      Пример: "v_1"

    ВОЗВРАЩАЕТ:
    - dict с распределением по классам для каждого сплита

    ВАЖНЫЕ ЗАМЕЧАНИЯ:
    - Если какой-то класс отсутствует в сплите, это проблема разбиения — сообщи об этом
    """

    @tool_response(DATASETS_URL)
    def _run(self, dataset_id: str, version_id: str) -> str:
        url = f"{DATASETS_URL}/datasets/{dataset_id}/versions/{version_id}/splits/distribution"
        return get_json(url) # type: ignore[return-value]  Декоратор преобразет ответ в str

    async def _arun(self, dataset_id: str, version_id: str) -> str:
        return await asyncio.to_thread(self._run, dataset_id, version_id)


class GetDatasetImageSizeStatsTool(BaseTool):
    """Инструмент для получения статистики размеров изображений по сплитам версии датасета"""

    name: str = "GetDatasetImageSizeStats"
    description: str = """
    НАЗНАЧЕНИЕ: Получить статистику размеров изображений в каждом сплите.

    КОГДА ИСПОЛЬЗОВАТЬ:
    - Для выбора размера входа модели (input size)
    - Для настройки resize/crop в препроцессинге
    - Когда нужно понять, насколько разнородны размеры изображений

    ВХОДНЫЕ ДАННЫЕ:
    - dataset_id (str): ID датасета.
      Пример: "human_face_emotions"
    - version_id (str): ID версии датасета.
      Пример: "v_1"

    ВОЗВРАЩАЕТ:
    - dict со статистикой размеров изображений по сплитам

    ВАЖНЫЕ ЗАМЕЧАНИЯ:
    - Учитывай эту статистику при подборе параметров препроцессинга
    """

    @tool_response(DATASETS_URL)
    def _run(self, dataset_id: str, version_id: str) -> str:
        url = f"{DATASETS_URL}/datasets/{dataset_id}/versions/{version_id}/splits/size-stats"
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

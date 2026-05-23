import asyncio
import aiohttp
from typing import Dict, Any, List
from crewai.tools import BaseTool
from pydantic import Field

from .utils import get_json, handle_errors
from app.logs import get_logger

logger = get_logger(__name__)

class GetModelDetailsTool(BaseTool):
    """Инструмент для получения полной информации об одной ML модели по её ID"""

    name: str = "GetModelDetails"
    description: str = """
    НАЗНАЧЕНИЕ: Получить полную информацию об ОДНОЙ ML модели по её ID.

    КОГДА ИСПОЛЬЗОВАТЬ:
    - Когда нужно узнать детали конкретной модели
    - Когда у вас есть один ID модели

    ВХОДНЫЕ ДАННЫЕ:
    - model_id (str): Уникальный идентификатор модели.
      Пример: "f0536964-7950-4087-aa93-91bd50d835be"

    ВОЗВРАЩАЕТ:
    - dict с информацией о модели
    """

    @handle_errors
    def _run(self, model_id: str) -> Dict[str, Any]:
        """Выполнение инструмента"""
        return get_json(f"/models/{model_id}")

    async def _arun(self, model_id: str) -> Dict[str, Any]:
        """Асинхронная версия"""
        return get_json(f"/models/{model_id}")


class GetMultipleModelsDetailsTool(BaseTool):
    """Инструмент для получения информации о нескольких ML моделях по списку ID"""

    name: str = "GetMultipleModelsDetails"
    description: str = """
    НАЗНАЧЕНИЕ: Получить информацию о НЕСКОЛЬКИХ ML моделях по списку ID.

    КОГДА ИСПОЛЬЗОВАТЬ:
    - Когда нужно сравнить несколько моделей
    - Когда нужно получить информацию о 2+ моделях за один вызов
    - Основной инструмент для анализа и сравнения моделей

    ВХОДНЫЕ ДАННЫЕ:
    - model_ids (List[str]): Список ID моделей.
      Пример: ["f0536964-7950-4087-aa93-91bd50d835be", "9b5cddc4-6992-471d-b423-4de7be7c2b91"]

    ВОЗВРАЩАЕТ:
    - dict, где ключ = ID модели, значение = информация о модели:
        {
            "model_id_1": { ... },
            "model_id_2": { ... }
        }

    ВАЖНО:
    - Этот инструмент предпочтительнее, чем многократный вызов GetModelDetails
    - Используй его для сравнения моделей
    """

    @handle_errors
    def _run(self, model_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Выполнение инструмента"""
        info_models = {}

        for model_id in model_ids:
            info_models[model_id] = get_json(f"/models/{model_id}")

        return info_models

    async def _arun(self, model_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Асинхронная версия с параллельными запросами"""        

        async with aiohttp.ClientSession() as session:
            tasks = [self.__fetch_model(session, model_id) for model_id in model_ids]
            results = await asyncio.gather(*tasks)
            return dict(results)

    async def __fetch_model(self, session, model_id: str):
        async with session.get(f"/models/{model_id}") as response:
            return model_id, await response.json()
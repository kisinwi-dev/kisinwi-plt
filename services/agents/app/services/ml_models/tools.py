import asyncio
from typing import Dict, Any
from crewai.tools import BaseTool

from ..utils import get_json, tool_response
from app.config import config_url
from app.logs import get_logger

logger = get_logger(__name__)

ML_MODELS_URL = config_url.ML_MODELS['url']

class GetModelDetailsTool(BaseTool):
    """Инструмент для получения полной информации об одной ML модели по её ID"""

    name: str = "GetModelDetails"
    description: str = """
    НАЗНАЧЕНИЕ: Получить полную информацию об одной ML модели по её ID.

    КОГДА ИСПОЛЬЗОВАТЬ:
    - Когда нужно узнать детали конкретной модели
    - Когда у вас есть один ID модели

    ВХОДНЫЕ ДАННЫЕ:
    - model_id (str): Уникальный идентификатор модели.
      Пример: "f0536964-7950-4087-aa93-91bd50d835be"

    ВОЗВРАЩАЕТ:
    - dict с информацией о модели
    """

    @tool_response(ML_MODELS_URL)
    def _run(self, model_id: str) -> str:
        url = f"{ML_MODELS_URL}/models/{model_id}"
        return get_json(url) # type: ignore[return-value]  Декоратор преобразет ответ в str

    async def _arun(self, model_id: str) -> str:
        return await asyncio.to_thread(self._run, model_id)

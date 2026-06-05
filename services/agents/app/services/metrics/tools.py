from typing import Any, Dict
from crewai.tools import BaseTool

from ..utils import get_json, tool_response
from app.config import config_url
from app.logs import get_logger

logger = get_logger(__name__)

METRICS_URL = config_url.METRICS['url']

class GetMetricsForModelTool(BaseTool):
    """Инструмент для получения полного списка метрик ML модели по её ID"""

    name: str = "GetMetricsForModel"
    description: str = """
    НАЗНАЧЕНИЕ: Получить полную информацию о метриках ML модели по её ID.

    КОГДА ИСПОЛЬЗОВАТЬ:
    - Когда нужно оценить качество конкретной модели
    - Для сравнения нескольких моделей между собой
    - Для определения лучшей модели по метрикам
    - Перед рекомендацией модели к использованию в продакшене
    - Когда нужно понять, переобучилась ли модель (сравнение train/val метрик)

    ВХОДНЫЕ ДАННЫЕ:
    - model_id (str): Уникальный идентификатор модели в системе.
      Пример: "f0536964-7950-4087-aa93-91bd50d835be"

    ВОЗВРАЩАЕТ:
    - Словарь метрик и список метрик от 1 эпохи обучения к финальной

    ВАЖНЫЕ ЗАМЕЧАНИЯ:
    - Перед использованием вызови инструмент DoesModelHaveMetrics, что бы проверить наличие модели
    """

    @tool_response(METRICS_URL)
    def _run(self, model_id: str) -> str:
        return get_json(f"{METRICS_URL}/models/{model_id}")

    async def _arun(self, model_id: str) -> Dict[str, Any]:
        return get_json(f"{METRICS_URL}/models/{model_id}")


class DoesModelHaveMetricsTool(BaseTool):
    """Инструмент для проверки существования метрик у ML модели"""
    
    name: str = "DoesModelHaveMetrics"
    description: str = """
    НАЗНАЧЕНИЕ: Проверить, существуют ли метрики для указанной ML модели.

    КОГДА ИСПОЛЬЗОВАТЬ:
    - Перед вызовом GetMetricsForModel чтобы убедиться, что метрики есть
    - Когда нужно отфильтровать модели без метрик
    - Для валидации, можно ли сравнивать модель с другими
    - При построении отчётов, чтобы не запрашивать метрики у сырых моделей

    ВХОДНЫЕ ДАННЫЕ:
    - model_id (str): Уникальный идентификатор модели в системе.
      Пример: "f0536964-7950-4087-aa93-91bd50d835be"

    ВОЗВРАЩАЕТ:
    - bool: True если метрики существуют, False если отсутствуют

    ВАЖНЫЕ ЗАМЕЧАНИЯ:
    - Всегда проверяй существование метрик перед вызовом GetMetricsForModel
    - Если модель не найдена, вернётся False
    """

    @tool_response(METRICS_URL)
    def _run(self, model_id: str) -> str:
        data = get_json(f"{METRICS_URL}/models/{model_id}/exists")
        return data.get("exists", False)

    async def _arun(self, model_id: str) -> bool:
        data = get_json(f"{METRICS_URL}/models/{model_id}/exists")
        return data.get("exists", False)
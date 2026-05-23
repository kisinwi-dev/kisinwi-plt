from crewai.tools import BaseTool
from pydantic import Field

from .utils import get_json, handle_errors
from app.logs import get_logger

logger = get_logger(__name__)


class GetMetricsForModelTool(BaseTool):
    """Инструмент для получения метрик качества ML модели по её ID"""

    name: str = "GetMetricsForModel"
    description: str = """
    НАЗНАЧЕНИЕ: Получить метрики качества ML модели по её ID.

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
    - Cписок метрик модели

    ВАЖНЫЕ ЗАМЕЧАНИЯ:
    - Если метрики отсутствуют для модели, вернётся пустой список []
    """

    @handle_errors
    def _run(self, model_id: str):
        """Выполнение инструмента"""
        return get_json(f"/models/{model_id}")

    async def _arun(self, model_id: str):
        """Асинхронная версия"""
        return get_json(f"/models/{model_id}")


class DoesModelHaveMetricsTool(BaseTool):
    """Инструмент для проверки существования метрик у ML модели"""
    
    name: str = "DoesModelHaveMetricsTool"
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

    @handle_errors
    def _run(self, model_id: str):
        """Выполнение инструмента"""
        return get_json(f"/models/{model_id}/exists")

    async def _arun(self, model_id: str):
        """Асинхронная версия"""
        return get_json(f"/models/{model_id}/exists")
from typing import List, Dict, Any
from crewai.tools import tool

from .utils import get_json, handle_errors
from app.logs import get_logger

logger = get_logger(__name__)

@tool("GetMetricsForModel")
@handle_errors
def get_metrics(   
    model_id: str
) -> List[Dict[str, Any]]:
    """
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
    - dict с метриками:

    ПРИМЕР ВЫЗОВА:
    get_metrics("f0536964-7950-4087-aa93-91bd50d835be")

    ВАЖНЫЕ ЗАМЕЧАНИЯ:
    - Если метрики отсутствуют для модели, вернётся пустой словарь {}
    - При ошибке получения метрик вернётся {"error": "текст ошибки"}
    """
    logger.info(f"🔧 GetMetricsForModel вызван для модели: {model_id}")
    json = get_json(f"/models/{model_id}")
    return json["metrics"]
    
@tool("MetricsForModelExists")
@handle_errors
def model_metrics_exists(
    model_id: str
) -> bool:
    """
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

    ПРИМЕР ВЫЗОВА:
    model_metrics_exists("f0536964-7950-4087-aa93-91bd50d835be")

    ОТВЕТ: True или False

    ВАЖНЫЕ ЗАМЕЧАНИЯ:
    - Всегда проверяй существование метрик перед вызовом GetMetricsForModel
    - Если модель не найдена, вернётся False
    """
    logger.info(f"🔧 MetricsForModelExists вызван для модели: {model_id}")
    json = get_json(f"/models/{model_id}/exists")
    if json["exists"] is True:
        return True
    else:
        return False
    
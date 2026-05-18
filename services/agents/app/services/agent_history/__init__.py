from crewai.tools import tool

from .utils import (
    get_json, handle_errors, health
)
from app.logs import get_logger

logger = get_logger(__name__)

@tool("GetAgentHistoryInfo")
@handle_errors
def get_agent_history(discussion_id: str) -> dict:
    """
    НАЗНАЧЕНИЕ: Получить полную информацию об ответах агентов по ID диалога.

    КОГДА ИСПОЛЬЗОВАТЬ:
    - Когда нужно узнать историю рассуждений о обучения

    ВХОДНЫЕ ДАННЫЕ:
    - discussion_id (str): Уникальный идентификатор диалога.
      Пример: "8287449d-e15e-4d72-bc9b-21914ed75787"

    ПРИМЕР ВЫЗОВА:
    get_agent_history("8287449d-e15e-4d72-bc9b-21914ed75787")

    ВАЖНЫЕ ЗАМЕЧАНИЯ:
    - Используй этот инструмент перед работой с конкретными версиями
    """
    logger.info(f"🔧 GetAgentHistoryInfo вызван для дискуссия: {discussion_id}")
    return get_json(f"/discussions/{discussion_id}")
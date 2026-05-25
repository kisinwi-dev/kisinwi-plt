from typing import Dict, Any
from crewai.tools import BaseTool

from ..utils import (
    get_json, handle_errors
)
from app.config import config_url
from app.logs import get_logger

logger = get_logger(__name__)

class GetAgentHistoryTool(BaseTool):
    """Инструмент для получения истории работы агентов по ID диалога"""

    name: str = "GetAgentHistory"
    description: str = """
    НАЗНАЧЕНИЕ: Получить полную информацию действий агентов ID диалога.

    КОГДА ИСПОЛЬЗОВАТЬ:
    - Когда нужно узнать историю рассуждений
    - Когда нужно понять, какие инструменты уже вызывались
    - Когда нужно проанализировать предыдущие ответы

    ВХОДНЫЕ ДАННЫЕ:
    - discussion_id (str): Уникальный идентификатор диалога.
      Пример: "8287449d-e15e-4d72-bc9b-21914ed75787"
    """

    @handle_errors(config_url.AGENT_HISTORY)
    def _run(self, discussion_id: str) -> Dict[str, Any]:
        return get_json(f"{config_url.AGENT_HISTORY}/discussions/{discussion_id}")

    async def _arun(self, discussion_id: str) -> Dict[str, Any]:
        return get_json(f"{config_url.AGENT_HISTORY}/discussions/{discussion_id}")
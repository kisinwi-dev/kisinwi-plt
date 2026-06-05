from typing import Dict, Any
from crewai.tools import BaseTool

from ..utils import (
    get_json, handle_errors
)
from app.config import config_url
from app.logs import get_logger

logger = get_logger(__name__)

AGENT_HISTORY = config_url.AGENT_HISTORY['url']

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

    ВОЗВРАЩАЕТ:
    - dict с полями meta (метаданные диалога), responses (ответы агентов)
      и system_messages (системные сообщения)
    """

    @handle_errors(AGENT_HISTORY)
    def _run(self, discussion_id: str) -> Dict[str, Any]:
        return self._collect_history(discussion_id)

    async def _arun(self, discussion_id: str) -> Dict[str, Any]:
        return self._collect_history(discussion_id)

    def _collect_history(self, discussion_id: str) -> Dict[str, Any]:
        # Единого GET /discussions/{id} в сервисе истории нет — собираем диалог
        # из доступных эндпоинтов: meta + responses + system_messages.
        base = f"{AGENT_HISTORY}/discussions/{discussion_id}"
        return {
            "meta": get_json(f"{base}/meta"),
            "responses": get_json(f"{base}/responses"),
            "system_messages": get_json(f"{base}/system_messages"),
        }
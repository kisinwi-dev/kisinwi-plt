from typing import Dict, Any
from pydantic import Field
from crewai.tools import BaseTool

from .utils import (
    get_json, handle_errors
)
from app.logs import get_logger

logger = get_logger(__name__)

class GetAgentHistoryTool(BaseTool):
    """Инструмент для получения истории ответов агентов по ID диалога"""

    name: str = "GetAgentHistory"
    description: str = """
    НАЗНАЧЕНИЕ: Получить полную информацию об ответах агентов по ID диалога.

    КОГДА ИСПОЛЬЗОВАТЬ:
    - Когда нужно узнать историю рассуждений

    ВХОДНЫЕ ДАННЫЕ:
    - discussion_id (str): Уникальный идентификатор диалога.
      Пример: "8287449d-e15e-4d72-bc9b-21914ed75787"
    """

    @handle_errors
    def _run(self, discussion_id: str) -> Dict[str, Any]:
        """
        Выполнение инструмента.

        Args:
            discussion_id: ID диалога

        Returns:
            Dict[str, Any]: Информация об ответах агентов
        """
        return get_json(f"/discussions/{discussion_id}")

    async def _arun(self, discussion_id: str) -> Dict[str, Any]:
        """
        Асинхронная версия
        """
        return get_json(f"/discussions/{discussion_id}")
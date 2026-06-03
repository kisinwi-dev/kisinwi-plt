import requests
from typing import Any, Dict, Optional

from app.config import config_url
from app.logs import get_logger
from app.core.memory import discussion_context

logger = get_logger(__name__)


class AgentHistoryClient:
    """Клиент для работы с историей агентов"""

    def __init__(self):
        self.base_url = config_url.AGENT_HISTORY['url']

    def _make_discussion_request(
        self, 
        endpoint: str, 
        data: Dict[str, Any]
    ) -> bool:
        """
        Метод отправки запросов

        Args:
            endpoint: Endpoint для запроса
            data: Данные для отправки

        Returns:
            bool: Успешность операции
        """

        discussion_id = discussion_context.get()

        url = f"{self.base_url}/discussions/{discussion_id}/{endpoint}"

        try:
            response = requests.post(url, json=data)

            if response.status_code == 201:
                logger.debug(f"✅ Событие сохранено в историю discussion_id=`{discussion_id}`")
                return True
            else:
                logger.error(f"Ошибка отправки: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"Непредвиденная ошибка: {str(e)}")
            return False

    def _add_agent(
        self, 
        response_id: str,
        agent_role: str,
        text: str,
        status: str
    ) -> bool:
        """
        Добавление ответа агента в историю
        """
        data = {
            "response_id": response_id,
            "status": status,
            "agent_role": agent_role,
            "text": text
        }
        return self._make_discussion_request("responses", data)

    def agent_start(
        self, 
        response_id: str,
        agent_role: str,
        text: str,
    ) -> bool:
        """
        Добавление агента при инциализации первого

        Args:
            response_id: Id ответа,
            agent_role: Роль агента,
            text: Текст от агента,
        """
        return self._add_agent(
            response_id,
            agent_role,
            text,
            status="IN PROGRESS"
        )

    def agent_succeed(
        self, 
        response_id: str,
        agent_role: str,
        text: str,
    ) -> bool:
        """
        Добавление результатов работы агента

        Args:
            response_id: Id ответа,
            agent_role: Роль агента,
            text: Текст от агента,
        """
        return self._add_agent(
            response_id,
            agent_role,
            text,
            status="SUCCEED"
        )

    def _add_tool(
        self,
        id: str,
        agent_role: str,
        name: str,
        message: str,
        status: str,
        input_args: Optional[Dict[str, Any]] = None,
        output: Optional[Any] = None,
        duration_ms: Optional[float] = None,
        error_traceback: Optional[str] = None,
    ) -> bool:
        """Добавить вызов инструмента"""

        data: Dict[str, Any] = {
            "id": id,
            "agent_role": agent_role,
            "name": name,
            "status": status,
            "message": message,
        }

        if input_args is not None:
            data["input_args"] = input_args
        if output is not None:
            data["output"] = output
        if duration_ms is not None:
            data["duration_ms"] = duration_ms
        if error_traceback is not None:
            data["error_traceback"] = error_traceback

        return self._make_discussion_request("tool", data)

    def tool_start(
        self,
        id: str,
        agent_role: str,
        name: str,
        message: str | None,
        input_args: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Добавить историю инструмента"""

        message = message if message else "Нет информации"

        return self._add_tool(
            id,
            agent_role,
            name,
            message,
            status="IN PROGRESS",
            input_args=input_args,
        )

    def tool_succed(
        self,
        id: str,
        agent_role: str,
        name: str,
        message: str,
        output: Optional[Any] = None,
        duration_ms: Optional[float] = None,
    ) -> bool:
        """Вывести результат работы инструмента"""

        return self._add_tool(
            id,
            agent_role,
            name,
            message,
            status="SUCCEED",
            output=output,
            duration_ms=duration_ms,
        )

    def tool_error(
        self,
        id: str,
        agent_role: str,
        name: str,
        message: str,
        error_traceback: Optional[str] = None,
        duration_ms: Optional[float] = None,
    ) -> bool:
        """Вывести ошибку инструмента"""

        return self._add_tool(
            id,
            agent_role,
            name,
            message,
            status="ERROR",
            error_traceback=error_traceback,
            duration_ms=duration_ms,
        )


    def _add_system_message(
        self,
        type_: str,
        message: str,
    ) -> bool:
        """
        Добавление системного сообщения в историю

        Args:
            type_: Тип сообщения
            message: Текст сообщения
        """
        data = {
            "type_": type_,
            "message": message
        }

        return self._make_discussion_request("system_messages", data)

    def info(self, message: str) -> bool:
        """Добавить информационное сообщение"""
        return self._add_system_message("INFO", message)

    def warning(self, message: str) -> bool:
        """Добавить предупреждение"""
        return self._add_system_message("WARNING", message)

    def error(self, message: str) -> bool:
        """Добавить ошибку"""
        return self._add_system_message("ERROR", message)


agent_history_client = AgentHistoryClient()
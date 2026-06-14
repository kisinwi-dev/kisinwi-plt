import requests
from typing import Any, Dict, Optional

from app.config import config_url
from app.core.memory import discussion_context
from app.logs import get_logger

logger = get_logger(__name__)


class AgentHistoryClient:
    """Клиент для работы с историей агентов"""

    def __init__(self):
        self.base_url = config_url.AGENT_HISTORY['url']

    def _request(
        self,
        method: str,
        url: str,
        json: Dict[str, Any],
        *,
        ok_status: int,
        success_msg: str,
        error_label: str,
    ) -> bool:
        """
        Отправка fire-and-forget запроса в историю агентов.

        История не должна ронять пайплайн, поэтому любая ошибка логируется и
        возвращается False (без проброса исключения).
        """
        try:
            response = requests.request(method, url, json=json)
            if response.status_code == ok_status:
                logger.debug(success_msg)
                return True
            logger.error(f"{error_label}: {response.status_code} - {response.text}")
            return False
        except Exception as e:
            logger.error(f"{error_label} (непредвиденная ошибка): {e}")
            return False

    def create_discussion(
        self,
        discussion_id: str,
        pipeline: str,
        agent_roles: list[str],
        title: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> bool:
        """Создать дискуссию в agent_history"""
        url = f"{self.base_url}/discussions"
        data: Dict[str, Any] = {
            "discussion_id": discussion_id,
            "pipeline": pipeline,
            "agent_roles": agent_roles,
        }
        if title is not None:
            data["title"] = title
        if tags is not None:
            data["tags"] = tags
        return self._request(
            "POST", url, data,
            ok_status=201,
            success_msg=f"✅ Дискуссия создана discussion_id=`{discussion_id}`",
            error_label="Ошибка создания дискуссии",
        )

    def update_discussion_meta(
        self,
        discussion_id: str,
        status: str,
    ) -> bool:
        """Обновить метаданные дискуссии (например, статус)"""
        url = f"{self.base_url}/discussions/{discussion_id}/meta"
        return self._request(
            "PATCH", url, {"status": status},
            ok_status=200,
            success_msg=f"✅ Статус дискуссии обновлён: {status}",
            error_label="Ошибка обновления мета",
        )

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

        return self._request(
            "POST", url, data,
            ok_status=201,
            success_msg=f"✅ Событие сохранено в историю discussion_id=`{discussion_id}`",
            error_label="Ошибка отправки",
        )

    def _add_agent(
        self,
        response_id: str,
        agent_role: str,
        text: str,
        status: str,
        duration_ms: Optional[float] = None,
        model: Optional[str] = None,
        task_name: Optional[str] = None,
        iteration: Optional[int] = None,
    ) -> bool:
        """Добавление ответа агента в историю"""
        data: Dict[str, Any] = {
            "response_id": response_id,
            "status": status,
            "agent_role": agent_role,
            "text": text,
        }
        if duration_ms is not None:
            data["duration_ms"] = duration_ms
        if model is not None:
            data["model"] = model
        if task_name is not None:
            data["task_name"] = task_name
        if iteration is not None:
            data["iteration"] = iteration
        return self._make_discussion_request("responses", data)

    def agent_start(
        self,
        response_id: str,
        agent_role: str,
        text: str,
        model: Optional[str] = None,
        task_name: Optional[str] = None,
        iteration: Optional[int] = None,
    ) -> bool:
        return self._add_agent(
            response_id, agent_role, text, status="IN PROGRESS",
            model=model, task_name=task_name, iteration=iteration,
        )

    def agent_succeed(
        self,
        response_id: str,
        agent_role: str,
        text: str,
        duration_ms: Optional[float] = None,
        model: Optional[str] = None,
        task_name: Optional[str] = None,
        iteration: Optional[int] = None,
    ) -> bool:
        return self._add_agent(
            response_id, agent_role, text, status="SUCCEED",
            duration_ms=duration_ms, model=model, task_name=task_name, iteration=iteration,
        )

    def agent_error(
        self,
        response_id: str,
        agent_role: str,
        text: str,
        duration_ms: Optional[float] = None,
        model: Optional[str] = None,
        task_name: Optional[str] = None,
        iteration: Optional[int] = None,
    ) -> bool:
        """Отметить ответ агента как завершившийся с ошибкой (перезаписывает IN PROGRESS)."""
        return self._add_agent(
            response_id, agent_role, text, status="ERROR",
            duration_ms=duration_ms, model=model, task_name=task_name, iteration=iteration,
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
        response_id: Optional[str] = None,
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
        if response_id is not None:
            data["response_id"] = response_id

        return self._make_discussion_request("tool", data)

    def tool_start(
        self,
        id: str,
        agent_role: str,
        name: str,
        message: str | None,
        input_args: Optional[Dict[str, Any]] = None,
        response_id: Optional[str] = None,
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
            response_id=response_id,
        )

    def tool_succeeded(
        self,
        id: str,
        agent_role: str,
        name: str,
        message: str,
        output: Optional[Any] = None,
        duration_ms: Optional[float] = None,
        response_id: Optional[str] = None,
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
            response_id=response_id,
        )

    def tool_error(
        self,
        id: str,
        agent_role: str,
        name: str,
        message: str,
        error_traceback: Optional[str] = None,
        duration_ms: Optional[float] = None,
        response_id: Optional[str] = None,
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
            response_id=response_id,
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
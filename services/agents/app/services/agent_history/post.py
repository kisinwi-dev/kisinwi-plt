import requests
from typing import Any, Dict, Optional
from enum import Enum
from uuid import uuid4

from app.config import config_url
from app.logs import get_logger
from app.core.memory import discussion_context

logger = get_logger(__name__)


class SystemMessageType(Enum):
    """Типы системных сообщений"""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    TOOL_CALL = "TOOL_CALL"
    TOOL_RESULT = "TOOL_RESULT"
    AGENT_START = "AGENT_START"


class AgentHistoryClient:
    """Клиент для работы с историей агентов"""

    def __init__(self):
        self.base_url = config_url.AGENT_HISTORY        
    
    def _make_request(
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
                logger.info(f"✅ Событие '{data['type_']}' сохранено в историю discussion_id=`{discussion_id}`")
                return True
            else:
                logger.error(f"Ошибка отправки: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Непредвиденная ошибка: {e}")
            return False
    
    def add_response(
        self, 
        response_id: str, 
        agent_role: str, 
        agent_response: str
    ) -> bool:
        """
        Добавление ответа агента в историю
        
        Args:
            response_id: ID ответа
            agent_role: Роль агента
            agent_response: Полученный ответ от агента
        """
        data = {
            "type_": "AGENT_RESPONSE",
            "response_id": response_id,
            "agent_role": agent_role,
            "text": agent_response
        }
        return self._make_request("responses", data)
    
    def add_system_message(
        self,
        type_: SystemMessageType,
        message: str,
    ) -> bool:
        """
        Добавление системного сообщения в историю
        
        Args:
            type_: Тип сообщения
            message: Текст сообщения
        """
        data = {
            "type_": type_.value if isinstance(type_, SystemMessageType) else type_,
            "message": message
        }
            
        return self._make_request("system_messages", data)
        
    def info(self, message: str) -> bool:
        """Добавить информационное сообщение"""
        return self.add_system_message(SystemMessageType.INFO, message)
    
    def warning(self, message: str) -> bool:
        """Добавить предупреждение"""
        return self.add_system_message(SystemMessageType.WARNING, message)
    
    def error(self, message: str) -> bool:
        """Добавить ошибку"""
        return self.add_system_message(SystemMessageType.ERROR, message)
    
    def tool_call(
        self, 
        agent_role: str, 
        tool_name: str,
        tool_desc: str | None = None
    ) -> bool:
        """Добавить вызов инструмента"""
        msg =f"Агент {agent_role} вызвал инструмент {tool_name}"
        msg += f"\nОписание: {tool_desc if tool_desc else "нет описания"}"
        
        return self.add_system_message(
            SystemMessageType.TOOL_CALL,
            msg
        )
    
    def tool_result(
        self,
        tool_name: str,
        message: str | None = None
    ) -> bool:
        """Добавить результат работы инструмента"""
        msg = message or f"Инструмент {tool_name} закончил выполнение"
        return self.add_system_message(
            SystemMessageType.TOOL_RESULT,
            msg
        )
    
    def agent_start(
        self,
        agent_name: str,
    ) -> bool:
        """Добавить информацию о старте работы агента"""
        msg = f"Агент '{agent_name}' начал работу"
        return self.add_system_message(
            SystemMessageType.AGENT_START,
            msg
        )

agent_history_client = AgentHistoryClient()
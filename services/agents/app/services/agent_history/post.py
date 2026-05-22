import requests
from typing import Any, Dict

from app.config import config_url
from app.logs import get_logger
from app.core.memory import discussion_context

logger = get_logger(__name__)

def add_reponse_in_history(
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

    agent_info = {}
    agent_info["response_id"] = response_id
    agent_info["text"] = agent_response
    agent_info["agent_role"] = agent_role

    discussion_id = discussion_context.get()
    url = f"{config_url.AGENT_HISTORY}/discussions/{discussion_id}/responses"
    
    try:
        
        result = requests.post(
            url,
            json=agent_info
        )
        
        if result.status_code == 201:
            logger.info(f"✅ Ответ агента занесен в историю discussion_id=`{discussion_id}` response_id='{response_id}'")
            return True
        else:
            logger.error(f"Ошибка отправки: {result.status_code} - {result.text}")
            return False

    except Exception as e:
        logger.error(f"Непредвиденная ошибка: {e}")
        return False

def add_system_message(
    type_: str,
    message: str
) -> bool:
    """
    Добавление сообщения от системы в историю
    
    Args:
        type_: Тип сообщения
            На данный момент реализовано несколько типов:
            * "INFO"
            * "WARNING"
            * "ERROR"
            * "TOOL_CALL"
            * "TOOL_RESULT"
            * "AGENT_RESPONSE"
        message: Сообщение
    """

    message_info = {}
    message_info["type_"] = type_
    message_info["message"] = message

    discussion_id = discussion_context.get()
    url = f"{config_url.AGENT_HISTORY}/discussions/{discussion_id}/system_messages"
    
    try:
        
        result = requests.post(
            url,
            json=message_info
        )
        
        if result.status_code == 201:
            logger.info(f"✅ Запись действия занесена в историю discussion_id=`{discussion_id}`")
            return True
        else:
            logger.error(f"Ошибка отправки: {result.status_code} - {result.text}")
            return False

    except Exception as e:
        logger.error(f"Непредвиденная ошибка: {e}")
        return False
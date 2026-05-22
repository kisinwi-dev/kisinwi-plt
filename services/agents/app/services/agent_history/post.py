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
    discussion_id = discussion_context.get()

    agent_info = {}
    agent_info["response_id"] = response_id
    agent_info["text"] = agent_response
    agent_info["agent_role"] = agent_role

    return _post(discussion_id, agent_info)

def _post(
    discussion_id: str,
    agent_response: Dict[str, Any]
) -> bool:
    """Синхронная отправка ответа агента в сервис истории"""
    url = f"{config_url.AGENT_HISTORY}/discussions/{discussion_id}/responses"
    
    try:
        
        result = requests.post(
            url,
            json=agent_response
        )
        
        if result.status_code == 201:
            logger.info(f"✅ Ответ агента занесён в историю discussion_id=`{discussion_id}` response=`{agent_response["response_id"]}`")
            return True
        else:
            logger.error(f"Ошибка отправки: {result.status_code} - {result.text}")
            return False

    except Exception as e:
        logger.error(f"Непредвиденная ошибка: {e}")
        return False

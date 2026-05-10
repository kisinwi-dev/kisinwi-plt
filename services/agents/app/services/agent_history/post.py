import requests
from typing import Any, Dict
from crewai import Crew
from crewai.types.usage_metrics import UsageMetrics

from app.config import config_url
from app.logs import get_logger

logger = get_logger(__name__)

def add_reponse_in_history(
        crew: Crew,
        discussion_id: str,
        reponse: str
) -> bool:
    
    agent_info = {}

    response_id = str(crew.id)
    agent_info["text"] = reponse
    agent_info["agent_role"] = crew.agents[0].role

    _post(discussion_id, response_id, agent_info)

    return True

def _post(
    discussion_id: str,
    response_id: str, 
    agent_response: Dict[str, Any],
) -> bool:
    """
    Синхронная отправка ответа агента
    """
    url = f"{config_url.AGENT_HISTORY}/discussions/{discussion_id}/responses/{response_id}"
    
    try:
        
        result = requests.post(
            url,
            json=agent_response
        )
        
        if result.status_code == 201:
            logger.info(f"✅ Агент занесён в историю {agent_response["response_id"]}")
            return True
        else:
            logger.error(f"Ошибка отправки: {result.status_code} - {result.text}")
            return False

    except Exception as e:
        logger.error(f"Непредвиденная ошибка: {e}")
        return False

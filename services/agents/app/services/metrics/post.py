import requests
from typing import Any, Dict
from crewai import Crew
from crewai.types.usage_metrics import UsageMetrics

from app.config import config_url
from app.logs import get_logger

logger = get_logger(__name__)

def add_agent_in_metrics(
        crew: Crew
) -> bool:
    
    agent_info = {}

    agent_info["response_id"] = str(crew.id)
    agent_info["metrics"] = _token_metrics_to_dict(crew.usage_metrics)

    _post(agent_info)

    return True

def _post(
    agent_response: Dict[str, Any],
) -> bool:
    """
    Синхронная отправка ответа агента
    """
    url = f"{config_url.METRICS_URL}/agents/add"
    
    try:
        
        result = requests.post(
            url,
            json=agent_response
        )
        
        if result.status_code == 200:
            logger.info(f"✅ Метрики ответа {agent_response["response_id"]} отправлены в сервис метрик")
            return True
        else:
            logger.error(f"Ошибка отправки: {result.status_code} - {result.text}")
            return False

    except Exception as e:
        logger.error(f"Непредвиденная ошибка: {e}")
        return False


def _token_metrics_to_dict(
        metrics: UsageMetrics | None
) -> Dict[str, Any]:
    """Конвертирует UsageMetrics в словарь"""
    if metrics is None:
        return {"available": False}
    
    return {
        "available": True,
        "total_tokens": metrics.total_tokens,
        "prompt_tokens": metrics.prompt_tokens,
        "completion_tokens": metrics.completion_tokens,
        "successful_requests": metrics.successful_requests
    }
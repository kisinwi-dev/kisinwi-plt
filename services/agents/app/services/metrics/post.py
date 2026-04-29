import requests
from typing import Any, Dict
from crewai import Crew
from crewai.types.usage_metrics import UsageMetrics

from app.config import config_url
from app.logs import get_logger

logger = get_logger(__name__)

def add_agent_in_metrics(
        crew: Crew, 
        conversation_id: str
) -> bool:
    
    agent_info = {}

    for i in range(len(crew.agents)):

        agent = crew.agents[i]
        task = crew.tasks[i]

        name = agent.role
        tools = []

        if isinstance(agent.tools, list):
            for tool in agent.tools:
                tools.append(tool.name)

        agent_info["agent"] = {
            "name": name,
            "tools": tools
        }
        agent_info["conversation_id"] = conversation_id
        agent_info["response_id"] = str(crew.id)
        
        if task.output and hasattr(task.output, 'raw'):
            agent_info["out"] = task.output.raw
        else:
            agent_info["out"] = str(task.output)
        
        agent_info["metrics"] = _token_metrics_to_dict(crew.usage_metrics)

        _post(agent_info)

    return True

def _post(
    agent_response: Dict[str, Any],
) -> bool:
    """
    Синхронная отправка ответа агента
    """
    url = f"{config_url.METRICS_URL}/agents/response/add"
    
    try:
        
        result = requests.post(
            url,
            json=agent_response
        )
        
        if result.status_code == 200:
            logger.info(f"✅ Ответ агента {agent_response["response_id"]} отправлен в сервис метрик")
            return True
        else:
            logger.error(f"❌ Ошибка отправки: {result.status_code} - {result.text}")
            return False

    except Exception as e:
        logger.error(f"😡 Ошибка: {e}")
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
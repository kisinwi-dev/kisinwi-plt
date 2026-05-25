import requests
from app.config import config_url
from app.logs import get_logger

logger = get_logger(__name__)

def health() -> dict:
    """Проверка жизнеспособности сервиса"""
    try:
        response = requests.get(
            f"{config_url.AGENT_HISTORY}/info/health",
            timeout=30
        )        
        response.raise_for_status()
        
        return response.json()
    
    except requests.RequestException as e:
        logger.error(f"Ошибка HTTP при обращении к сервису хранения истории агентов: {e}")
        return {
            "status": "dead"
        }
    except Exception as e:
        logger.error(f"Ошибка при обращении к сервису хранения истории агентов: {e}")
        return {
            "status": "dead"
        }
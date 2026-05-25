import requests
from typing import List

from ..utils import get_json, handle_errors
from app.config import config_url
from app.logs import get_logger

logger = get_logger(__name__)

@handle_errors(config_url.DMS_URL)
def get_dataset_info_classes(dataset_id: str) -> List[str]:
    """
    Получить всю информацию о датасете по ID
    
    Args:
        dataset_id: Id датасета
        
    Returns:
        Dict с информацией о датасете или ошибкой
    """    
    return get_json(f"{config_url.DMS_URL}/api/datasets/{dataset_id}")["class_names"]

def health() -> dict:
    try:
        # Отправляем  запрос
        response = requests.get(
            f"{config_url.DMS_URL}/info/health",
            timeout=30
        )        
        response.raise_for_status()
        return response.json()
    
    except requests.RequestException as e:
        logger.error(f"Ошибка HTTP при обращении к сервису датасетов: {e}")
        return {
            "status": "dead"
        }
    except Exception as e:
        logger.error(f"Ошибка при обращении к сервису датасетов: {e}")
        return {
            "status": "dead"
        }

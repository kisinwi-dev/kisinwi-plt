import requests
from typing import List

from ..utils import get_json, handle_errors
from app.config import config_url
from app.logs import get_logger

logger = get_logger(__name__)

DATASETS_URL = config_url.DATASETS['url']

@handle_errors(DATASETS_URL)
def get_dataset_details(dataset_id: str) -> dict:
    """
    Получить полную информацию о датасете по ID

    Args:
        dataset_id: Id датасета

    Returns:
        Dict с информацией о датасете или ошибкой
    """
    return get_json(f"{DATASETS_URL}/datasets/{dataset_id}")

@handle_errors(DATASETS_URL)
def get_dataset_version_details(dataset_id: str, version_id: str) -> dict:
    """
    Получить информацию о версии датасета

    Args:
        dataset_id: Id датасета
        version_id: Id версии датасета

    Returns:
        Dict с информацией о версии или ошибкой
    """
    return get_json(f"{DATASETS_URL}/datasets/{dataset_id}/versions/{version_id}")

@handle_errors(DATASETS_URL)
def get_dataset_info_classes(dataset_id: str) -> List[str]:
    """
    Получить всю информацию о датасете по ID
    
    Args:
        dataset_id: Id датасета
        
    Returns:
        Dict с информацией о датасете или ошибкой
    """    
    return get_json(f"{DATASETS_URL}/datasets/{dataset_id}")["classes_names"]

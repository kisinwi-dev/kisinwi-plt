import requests
from functools import wraps

from app.config import config_url
from app.logs import get_logger

logger = get_logger(__name__)

def get_json(endpoint: str) -> dict:
    resp = requests.get(f"{config_url.DMS_URL}{endpoint}")
    resp.raise_for_status()
    return resp.json()

# __WARNING__ ТРЕБУЕТСЯ РЕАЛИЗОВАТЬ ТАКОЙ МЕТОД В СЕРВИСЕ С ДАТАСЕТАМИ
def get_sample_sizes_for_all_data(dataset_info: dict) -> dict:
    """
    Извлекает информацию о размерах выборки из полной информации о датасете.
    
    Args:
        dataset_info: Полный JSON с информацией о датасете
        
    Returns:
        Словарь с информацией о размерах выборки
    """
    
    result = {
        "version_id": dataset_info["version_id"],
        "total_samples": dataset_info["num_samples"],
        "splits": {}
    }
    
    # Извлекаем информацию по каждому сплиту
    for split_name, split_data in dataset_info["splits"].items():
        class_distribution = split_data["class_distribution"]
        
        # Общее количество в сплите
        total_in_split = sum(cls.get("count", 0) for cls in class_distribution)
        
        # Распределение по классам
        class_counts = {
            cls["class_name"]: cls["count"]
            for cls in class_distribution
        }
        
        result["splits"][split_name] = {
            "total": total_in_split,
            "classes": class_counts
        }
    
    return result


def handle_errors(func):
    """Декоратор для обработки ошибок API запросов"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.Timeout:
            error_str = f"Timeout: {func.__name__} не ответил..."
            logger.error(error_str)
            return {"ERROR": error_str}
        except requests.exceptions.ConnectionError:
            error_str = f"Connection error: Не удалось подключиться к {config_url.DMS_URL}"
            logger.error(error_str)
            return {"ERROR": error_str}
        except requests.exceptions.HTTPError as e:
            error_str = f"HTTP error {e.response.status_code}: {e.response.text}"
            logger.error(error_str)
            return {"ERROR": error_str}
        except Exception as e:
            error_str = f"Ошибка в {func.__name__}: {str(e)}"
            logger.error(error_str)
            return {"ERROR": error_str}
    return wrapper
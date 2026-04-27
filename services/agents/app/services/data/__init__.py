import requests
from functools import wraps
from crewai.tools import tool

from .tools import get_sample_sizes_fot_all_data
from app.config import config_url

def handle_errors(func):
    """Декоратор для обработки ошибок API запросов"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.Timeout:
            return {"ERROR": f"Timeout: {func.__name__} не ответил..."}
        except requests.exceptions.ConnectionError:
            return {"ERROR": f"Connection error: Не удалось подключиться к {config_url.DMS_URL}"}
        except requests.exceptions.HTTPError as e:
            return {"ERROR": f"HTTP error {e.response.status_code}: {e.response.text}"}
        except Exception as e:
            return {"ERROR": f"Ошибка в {func.__name__}: {str(e)}"}
    return wrapper


@tool("GetDatasetInfo")
@handle_errors
def get_dataset_info(dataset_id: str) -> dict:
    """Получить информацию о датасете по ID"""
    try:
        resp = requests.get(f"{config_url.DMS_URL}/api/datasets/{dataset_id}")
        data = resp.json()
        return data
    except Exception as e:
        return {"ERROR": f"Ошибка: {e}"}
    
@tool("GetDatasetVersionInfo")
@handle_errors
def get_version_info(
    dataset_id: str,
    version_id: str
) -> dict:
    """Получить информацию о версии датасета по ID"""
    try:
        resp = requests.get(f"{config_url.DMS_URL}/api/datasets/{dataset_id}/versions/{version_id}")
        data = resp.json()
        return data
    except Exception as e:
        return {"ERROR": f"Ошибка: {e}"}
    
@tool("GetDatasetVersionSplitInfo")
@handle_errors
def get_version_split_info(
    dataset_id: str,
    version_id: str
) -> dict:
    """Получить информацию о версии датасета по ID"""
    try:
        resp = requests.get(f"{config_url.DMS_URL}/api/datasets/{dataset_id}/versions/{version_id}")
        data = resp.json()

        return get_sample_sizes_fot_all_data(data)
    except Exception as e:
        return {"ERROR": f"Ошибка: {e}"}

@tool("ListDatasets")
@handle_errors
def list_datasets() -> dict:
    """Список всех датасетов"""
    try:
        resp = requests.get(f"{config_url.DMS_URL}/api/datasets/")
        datasets = resp.json()
        return datasets
    except Exception as e:
        return {"ERROR": f"Ошибка: {e}"}

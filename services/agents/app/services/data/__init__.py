import requests
from crewai.tools import tool

from .tools import get_sample_sizes_fot_all_data
from app.config import config_url

@tool("GetDatasetInfo")
def get_dataset_info(dataset_id: str) -> dict:
    """Получить информацию о датасете по ID"""
    try:
        resp = requests.get(f"{config_url.DMS_URL}/api/datasets/{dataset_id}")
        data = resp.json()
        return data
    except Exception as e:
        return {"ERROR": f"Ошибка: {e}"}
    
@tool("GetDatasetVersionInfo")
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
    
@tool("GetDatasetVersionInfo")
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
def list_datasets() -> dict:
    """Список всех датасетов"""
    try:
        resp = requests.get(f"{config_url.DMS_URL}/api/datasets/")
        datasets = resp.json()
        return datasets
    except Exception as e:
        return {"ERROR": f"Ошибка: {e}"}

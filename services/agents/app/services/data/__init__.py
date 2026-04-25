import requests
from crewai.tools import tool

from app.config import config_url

@tool("GetDatasetInfo")
def get_dataset_info(dataset_id: str) -> str:
    """Получить информацию о датасете по ID"""
    try:
        resp = requests.get(f"{config_url.DMS_URL}/api/datasets/{dataset_id}")
        data = resp.json()
        return data
    except Exception as e:
        return f"Ошибка: {e}"

@tool("ListDatasets")
def list_datasets() -> str:
    """Список всех датасетов"""
    try:
        resp = requests.get(f"{config_url.DMS_URL}/api/datasets/")
        datasets = resp.json()
        return "\n".join([f"- {ds['name']} ({ds['dataset_id']})" for ds in datasets])
    except Exception as e:
        return f"Ошибка: {e}"

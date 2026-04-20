import requests
from crewai.tools import tool

from app.config import config_url


class DataService:
    
    def __init__(self) -> None:
        self.URL = config_url.DMS_URL


    @tool("GetDatasetInfo")
    def get_dataset_info(self, dataset_id: str) -> str:
        """Получить информацию о датасете по ID"""
        try:
            resp = requests.get(f"{self.URL}/api/datasets/{dataset_id}")
            data = resp.json()
            return data
        except Exception as e:
            return f"Ошибка: {e}"

    @tool("ListDatasets")
    def list_datasets(self) -> str:
        """Список всех датасетов"""
        try:
            resp = requests.get(f"{self.URL}/api/datasets/")
            datasets = resp.json()
            return "\n".join([f"- {ds['name']} ({ds['dataset_id']})" for ds in datasets])
        except Exception as e:
            return f"Ошибка: {e}"

data_service = DataService()

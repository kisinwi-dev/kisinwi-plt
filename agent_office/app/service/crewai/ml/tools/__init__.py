import os
import requests
from crewai.tools import tool

TRAINER_URL = "http://" + os.getenv("TRAINER_DOMEN", "localhost:6200")

@tool("GetExampleJSONTrainer")
def get_example_run_config_trainer_json() -> str:
    """Получить пример JSON для запуска тренировки модели"""
    try:
        resp = requests.get(f"{TRAINER_URL}/get_example_config")
        data = resp.json()
        return data
    except Exception as e:
        return f"Ошибка: {e}"

@tool("GetAllAvailableModels")
def get_type_and_name_models() -> str:
    """Получить все имеющиеся модели в распоряжении. В зачениях json находится тип, а в списках имена"""
    try:
        resp = requests.get(f"{TRAINER_URL}/get_available_models")
        data = resp.json()
        return data
    except Exception as e:
        return f"Ошибка: {e}"

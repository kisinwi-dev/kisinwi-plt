import json
import requests
from typing import Dict

from app.logs import get_logger
from app.config import config_url

logger = get_logger(__name__)

class Tasker():
    def __init__(self) -> None:
        self.URL = config_url.TASKER_URL

    def post_task(
            self,
            json_data: Dict | str
    ):
        """Отправить JSON для запуска тренировки модели"""
        try:
            # Проверяем, что json_data не пустой
            if not json_data:
                return {"error": "JSON data is empty"}
            
            # Парсим JSON если это строка
            if isinstance(json_data, str):
                params = json.loads(json_data)
            else:
                params = json_data

            params = {"params": params}

            # Отправляем POST запрос
            response = requests.post(
                f"{self.URL}/tasks",
                json=params,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            # Проверяем статус ответа
            response.raise_for_status()
            
            return {
                "status": "success",
                "status_code": response.status_code,
                "response": response.json() if response.text else {"message": "Task created"}
            }
        
        except Exception as e:
            logger.error(f"Ошибка при отправке задачи в сервис задач: {e}")
            raise
        
tasker = Tasker()
import json
import requests
import re
import time
from enum import Enum
from typing import Dict

from app.logs import get_logger
from app.config import config_url

logger = get_logger(__name__)

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class Tasker():
    def __init__(self) -> None:
        self.URL = config_url.TASKER_URL
        self.session = requests.Session()

    def post_task(
            self,
            json_data: Dict | str
    ) -> str:
        """Отправить JSON для запуска тренировки модели"""
        try:
            # Проверяем, что json_data не пустой
            if not json_data:
                raise ValueError("JSON data cannot be empty")
            
            # Парсим JSON если это строка
            params = self._parse_in_json(json_data)

            # Отправляем POST запрос
            response = self.session.post(
                f"{self.URL}/tasks",
                json={"params": params},
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            # Проверяем статус ответа
            response.raise_for_status()
            task_id = response.json()["task_id"]
            logger.debug(f" Задач отправлена и имеет id={task_id}")
            
            return task_id
        
        except requests.RequestException as e:
            logger.error(f"Ошибка HTTP при отправке задачи: {e}")
            raise
        except Exception as e:
            logger.error(f"Ошибка при отправке задачи в сервис задач: {e}")
            raise
    
    def _parse_in_json(
        self, 
        data: Dict | str
    ) -> Dict:
        """Парсинг в JSON из строки или возврат словаря"""
        if isinstance(data, dict):
            return data
        
        # Очищаем строку от маркеров markdown
        data = re.sub(r'```json\s*\n?', '', data)
        cleaned = re.sub(r'```\s*\n?', '', data)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка парсинга JSON. Полученный текст:\n{cleaned}\nОшибка: {e}")
            raise ValueError(f"Invalid JSON format: {e}")

    def task_is_finish(self, task_id: str) -> TaskStatus:
        """Проверить, завершена ли задача"""
        try:
            response = self.session.get(
                f"{self.URL}/tasks/{task_id}",
            )
            response.raise_for_status()
            
            status = response.json().get("status")
            
            return TaskStatus(status)
        
        except requests.RequestException as e:
            logger.error(f"Ошибка при проверке статуса задачи {task_id}: {e}")
            raise requests.RequestException(e)
        
    def waiting_completed(self, task_id: str):
        """Ожидание завершения задачи"""
        while True:
            status = tasker.task_is_finish(task_id)
            if status == TaskStatus.COMPLETED:
                break
            elif status == TaskStatus.FAILED:
                logger.error(f"Задача {task_id} завершена с ошибкой")
                raise Exception(f"Задача {task_id} завершена с ошибкой")
            
            time.sleep(2)

    def close(self):
        self.session.close()

    def __exit__(self):
        self.close()

tasker = Tasker()
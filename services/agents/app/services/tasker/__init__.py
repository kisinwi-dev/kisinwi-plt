import requests
import time
from enum import Enum
from typing import Dict

from app.logs import get_logger
from app.config import config_url
from app.services.tools import parse_in_json

logger = get_logger(__name__)

class Tasker():
    def __init__(self) -> None:
        self.URL = config_url.TASKER_URL
        self.session = requests.Session()

    def task_training_create(
            self,
            task_name: str,
            model_id: str,
            discussion_id: str
    ) -> str:
        """Создание задачи для обучения"""
        try:
            
            data = {
                "task_name": task_name,
                "model_id": model_id,
                "discussion_id": discussion_id
            }

            # Парсим в JSON
            params = parse_in_json(data)

            # Отправляем POST запрос
            response = self.session.post(
                f"{self.URL}/tasks",
                json=params,
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

    def task_status(self, task_id: str) -> str:
        """Проверить, завершена ли задача"""
        try:
            response = self.session.get(
                f"{self.URL}/tasks/{task_id}",
            )
            response.raise_for_status()
            
            status = response.json().get("status")
            
            return status
        
        except requests.RequestException as e:
            logger.error(f"Ошибка при проверке статуса задачи {task_id}: {e}")
            raise requests.RequestException(e)
        
    def waiting_completed(self, task_id: str):
        """Ожидание завершения задачи"""
        while True:
            status = tasker.task_status(task_id)
            if status == "completed":
                break
            elif status == "failed":
                logger.error(f"Задача {task_id} завершена с ошибкой")
                raise Exception(f"Задача {task_id} завершена с ошибкой")
            
            time.sleep(2)

    def close(self):
        self.session.close()

    def __exit__(self):
        self.close()

tasker = Tasker()
import requests
import time
from typing import Tuple

from ..utils import parse_in_json
from app.logs import get_logger
from app.config import config_url

logger = get_logger(__name__)

class TaskerClient():
    def __init__(self) -> None:
        self.URL = config_url.TASKER['url']
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
            logger.debug(f"Задач отправлена и имеет id={task_id}")

            return task_id

        except requests.RequestException as e:
            logger.error(f"Ошибка HTTP при отправке задачи: {e}")
            raise
        except Exception as e:
            logger.error(f"Ошибка при отправке задачи в сервис задач: {e}")
            raise

    def get_task(self, task_id: str) -> dict:
        """Получение информации о задаче"""
        try:
            response = self.session.get(
                f"{self.URL}/tasks/{task_id}",
            )
            response.raise_for_status()
            task = response.json()
            return task

        except requests.RequestException as e:
            logger.error(f"Ошибка при проверке статуса задачи {task_id}: {e}")
            raise requests.RequestException(e)

    def waiting_completed(self, task_id: str) -> Tuple[bool, dict]:
        """
        Ожидание завершения задачи
        
        Returns:
            bool: true - задача завершена успешно, false - получена ошибка в процессе обучения
            dict: информация о задаче
        """
        while True:
            task = self.get_task(task_id)
            task_status = task.get("status", "failed")
            if task_status == "completed":
                return True, task
            elif task_status == "failed":
                logger.error(f"Задача `{task_id}` завершена с ошибкой")
                return False, task
            
            time.sleep(2)

    def close(self):
        self.session.close()

    def __exit__(self):
        self.close()

tasker_client = TaskerClient()
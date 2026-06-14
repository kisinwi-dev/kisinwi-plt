import requests
import time
from typing import Tuple

from ..utils import parse_in_json, BaseServiceClient
from app.core.cancellation import cancellation_registry
from app.core.memory import discussion_context
from app.logs import get_logger
from app.config import config_url

logger = get_logger(__name__)

class TaskerClient(BaseServiceClient):
    def __init__(self) -> None:
        super().__init__(config_url.TASKER['url'])

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
                timeout=30
            )
            response.raise_for_status()
            task = response.json()
            return task

        except requests.RequestException as e:
            logger.error(f"Ошибка при проверке статуса задачи {task_id}: {e}")
            raise requests.RequestException(e)

    def cancel_task(self, task_id: str) -> None:
        """Отмена задачи обучения (trainer останавливается на границе эпохи)."""
        try:
            response = self.session.post(
                f"{self.URL}/tasks/{task_id}/cancel",
                timeout=30
            )
            response.raise_for_status()
            logger.info(f"Запрошена отмена задачи обучения `{task_id}`")
        except requests.RequestException as e:
            logger.error(f"Ошибка при отмене задачи {task_id}: {e}")
            raise requests.RequestException(e)

    def waiting_completed(self, task_id: str) -> Tuple[bool, dict]:
        """
        Ожидание завершения задачи

        Пока ждём, отслеживаем запрос на остановку пайплайна: если пользователь
        остановил агентов во время обучения — один раз отменяем задачу в tasker
        и продолжаем ждать её терминальный статус (`cancelled`).

        Returns:
            bool: true - задача завершена успешно, false - получена ошибка в процессе обучения
            dict: информация о задаче
        """
        cancel_requested = False
        while True:
            task = self.get_task(task_id)
            task_status = task.get("status", "failed")
            if task_status == "completed":
                return True, task
            elif task_status in ("failed", "cancelled"):
                logger.error(f"Задача `{task_id}` завершена со статусом `{task_status}`")
                return False, task

            if not cancel_requested and discussion_context.is_set() \
                    and cancellation_registry.is_stop_requested(discussion_context.get()):
                logger.info(f"🟦 Остановка пайплайна: отменяем активную задачу обучения `{task_id}`")
                self.cancel_task(task_id)
                cancel_requested = True

            time.sleep(2)

tasker_client = TaskerClient()
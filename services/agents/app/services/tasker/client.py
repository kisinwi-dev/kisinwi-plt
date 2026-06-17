import requests
import time
from typing import Tuple

from ..utils import parse_in_json, BaseServiceClient
from app.logs import get_logger
from app.config import config_url, config_base_llm

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

    def cancel_discussion_tasks(self, discussion_id: str) -> None:
        """
        Отменить все активные (waiting/running) задачи обучения дискуссии.

        Вызывается при остановке пайплайна: процесс агентов уже убит, поэтому
        активное обучение в trainer нужно остановить отдельно, иначе оно
        продолжит жечь ресурсы. Сбои не пробрасываем — остановка не должна падать.
        """
        for status in ("running", "waiting"):
            try:
                resp = self.session.get(
                    f"{self.URL}/tasks",
                    params={"discussion_id": discussion_id, "status": status},
                    timeout=30,
                )
                resp.raise_for_status()
                tasks = resp.json().get("tasks", [])
            except requests.RequestException as e:
                logger.error(f"Не удалось получить {status}-задачи дискуссии {discussion_id}: {e}")
                continue
            for task in tasks:
                try:
                    self.cancel_task(task["id"])
                except requests.RequestException:
                    pass  # already logged in cancel_task

    def waiting_completed(self, task_id: str) -> Tuple[bool, dict]:
        """
        Ожидание завершения задачи

        Опрос устойчив к временным сетевым сбоям: единичный blip к tasker не
        должен ронять многочасовое обучение, которое продолжается в trainer.
        Сдаёмся только после TRAINING_POLL_MAX_CONSEC_ERRORS неудач подряд;
        успешный опрос сбрасывает счётчик. Верхнего лимита по времени нет —
        легитимное обучение может идти долго.

        Отмена обучения при остановке пайплайна делается снаружи (kill процесса
        агентов + tasker_client.cancel_discussion_tasks), здесь её не отслеживаем.

        Returns:
            bool: true - задача завершена успешно, false - ошибка/отмена/потеря связи
            dict: информация о задаче
        """
        consec_errors = 0
        max_errors = config_base_llm.TRAINING_POLL_MAX_CONSEC_ERRORS
        while True:
            try:
                task = self.get_task(task_id)
            except requests.RequestException as e:
                consec_errors += 1
                logger.warning(
                    f"🟧 Опрос задачи `{task_id}` не удался "
                    f"({consec_errors}/{max_errors}): {e}"
                )
                if consec_errors >= max_errors:
                    logger.error(
                        f"🟥 Потеряна связь с tasker при опросе задачи `{task_id}` "
                        f"после {max_errors} попыток подряд"
                    )
                    return False, {
                        "status": "failed",
                        "error_message": f"Потеряна связь с tasker: {e}",
                        "status_info": None,
                    }
                time.sleep(2)
                continue

            consec_errors = 0
            task_status = task.get("status", "failed")
            if task_status == "completed":
                return True, task
            elif task_status in ("failed", "cancelled"):
                logger.error(f"Задача `{task_id}` завершена со статусом `{task_status}`")
                return False, task

            time.sleep(2)

tasker_client = TaskerClient()
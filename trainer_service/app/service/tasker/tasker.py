import httpx
from typing import Optional

from app.logs import get_logger
from app.config import config_domain
from .shemas import Task, TaskStatus

logger = get_logger(__name__)

class Tasker_Service():
    def __init__(self) -> None:
        """Класс для общения с сервисом задач"""
        self._domen = config_domain.TASKER
        self._task_id = None
    
    def set_client(
            self,
            client: httpx.AsyncClient
        ) -> bool:
        self._client = client
        return True

    async def get_next_task(self) -> Optional[Task]:
        """Возвращает задачу или None при ошибке"""
        try:
            # Запрос к сервису
            resp = await self._client.post(f"{self._domen}/tasks/next")
            resp.raise_for_status()

            if resp.status_code == 204:
                logger.info("Нет доступных задач")
                return None

            task = Task(**resp.json())
            self.task_id = task.task_id

            return task
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP ошибка {e.response.status_code}")
            return None
        except httpx.ConnectError:
            logger.error("Сервис задач недоступен")
            return None
        
    async def update_status_task(
            self,
            progress: int | None = None,
            status: TaskStatus | None = None,
            description: str | None = None,
            task_id: str | None = None
    ) -> bool:
        """Обновляет статус задачи"""

        if task_id is None:
            task_id = self.task_id

        url = f"{self._domen}/tasks/{task_id}/status"
        json = {"status": status, "progress": progress, "description": description}
        
        try:
            await self._client.patch(url, json=json)
            return True
        except Exception as e:
            logger.error("Не удалось обновить статус задачи")
            logger.error(f"Ошибка: {e}")
            return False
        

tasker_service = Tasker_Service()
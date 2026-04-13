import httpx
from typing import Optional

from app.logs import get_logger
from app.config import config_domain
from .tasker_shemas import Task

logger = get_logger(__name__)

class Tasker_Service():
    def __init__(
            self,
            client: httpx.AsyncClient
    ) -> None:
        """Класс для общения с сервисом задач"""
        self._client = client
        self._domen = config_domain.TASKER
    
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

            return task
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP ошибка {e.response.status_code}")
            return None
        except httpx.ConnectError:
            logger.error("Сервис задач недоступен")
            return None
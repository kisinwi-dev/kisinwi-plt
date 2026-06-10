import httpx

from app.logs import get_logger
from app.config import config_services

logger = get_logger(__name__)

class TaskerClient():
    def __init__(self) -> None:
        """Класс для общения с сервисом задач"""
        self._domain = config_services.TASKER['url']
        self.task_id: str | None = None


    def set_client(
            self,
            client: httpx.AsyncClient
        ) -> bool:
        self._client = client
        return True

    async def get_next_task(self) -> dict | None:
        """Возвращает задачу или None при ошибке"""
        try:
            # Запрос к сервису
            resp = await self._client.get(f"{self._domain}/tasks/next")
            resp.raise_for_status()

            if resp.status_code == 204:
                return None

            task = resp.json()
            self.task_id = task["id"]

            return task
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP ошибка {e.response.status_code}")
            return None
        except httpx.ConnectError:
            logger.error("Сервис задач недоступен")
            return None
        
    async def update_status_task(
            self,
            status: str = "running",
            status_info: str | None = None, 
            percentages: int | None = None,
            error: str | None = None,
            task_id: str | None = None
    ) -> bool:
        """Обновляет статус задачи"""
        if task_id is None:
            task_id = self.task_id

        if task_id is None:
            logger.warning("Попытка обновить статус задачи без активной задачи")
            return False

        url = f"{self._domain}/tasks/{task_id}/status"
        data_json = {
            k: v for k, v in {
                "status": status,
                "status_info": status_info,
                "percentages": percentages,
                "error": error
            }.items() if v is not None
        }

        logger.debug(f"Отправка нового статуса задачи в сервис задач.\ntask_id:{task_id}\nData:{data_json}")

        try:
            await self._client.post(url, json=data_json)
            return True
        except Exception as e:
            logger.error("Не удалось обновить статус задачи")
            logger.error(f"Ошибка: {e}")
            return False
        

tasker_service = TaskerClient()
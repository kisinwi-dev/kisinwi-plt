import os
import httpx
import asyncio
from app.core import training_model
from app.logs import get_logger

TASKER_URL = "http://" + os.getenv("TASKER", "localhost:6110")

logger = get_logger(__name__)

async def worker_loop():
    async with httpx.AsyncClient(timeout=30.0) as client:
        logger.info("Сервис тренировки начал работу.")
        while True:
            # 1. Запрашиваем следующую задачу
            try:
                resp = await client.post(f"{TASKER_URL}/tasks/next")
            except Exception as e:
                logger.error("Ошибка подключения к сервису задач")
                continue
            
            # проверка полученного результата
            if resp.status_code == 204 or resp.text == "null" or not resp.content:
                logger.info("Задач нет. Отдыхаем.")
                await asyncio.sleep(1)
                continue

            task = resp.json()
            
            logger.debug("Полученный json:\n", task)

            task_id = task["task_id"]
            config = task["payload"]
            logger.info(f"Worker взял задачу: {task_id}")
            
            # 2. Обновляем статус: запущено
            await client.put(f"{TASKER_URL}/tasks/{task_id}/status", json={"status": "running", "progress": 0})
            
            try:
                
                # Процесс обучения
                training_model(task_id, config)

                # Завершение
                result = {"processed": config, "message": "success"}
                await client.put(f"{TASKER_URL}/tasks/{task_id}/status", 
                                    json={"status": "completed", "progress": 100, "result": result})
                logger.info(f"Задача {task_id} завершена")
                
            except Exception as e:
                await client.put(f"{TASKER_URL}/tasks/{task_id}/status",
                                 json={"status": "failed", "progress": 0, "result": {"error": str(e)}})
                logger.error(f"Ошибка {task_id}: {e}")

if __name__ == "__main__":
    asyncio.run(worker_loop())
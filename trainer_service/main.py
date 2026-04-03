import httpx
import asyncio
from app.core import training_model
from app.logs import get_logger

TASKER_URL = "http://localhost:1000"  # адрес таскера

logger = get_logger(__name__)

async def worker_loop():
    async with httpx.AsyncClient(timeout=30.0) as client:
        logger.info("Сервис тренировки начал работу.")  
        while True:
            # 1. Запрашиваем следующую задачу
            resp = await client.post(f"{TASKER_URL}/tasks/next")
            if resp.status_code == 204 or resp.status_code == 404 or resp.text == "null" or not resp.content:
                logger.info("Задач нет. Отдыхаем.")
                await asyncio.sleep(1)
                continue

            task = resp.json()
            if not task:
                await asyncio.sleep(1)
                continue
            
            task_id = task["task_id"]
            payload = task["payload"]
            print(f"Worker: взял задачу {task_id}")
            
            # 2. Обновляем статус: запущено
            await client.put(f"{TASKER_URL}/tasks/{task_id}/status", json={"status": "running", "progress": 0})
            
            try:
                
                # Процесс обучения
                training_model(task_id, payload["params_train"])

                # Завершение
                result = {"processed": payload, "message": "success"}
                await client.put(f"{TASKER_URL}/tasks/{task_id}/status", 
                                    json={"status": "completed", "progress": 100, "result": result})
                print(f"Worker: задача {task_id} завершена")
                
            except Exception as e:
                await client.put(f"{TASKER_URL}/tasks/{task_id}/status",
                                 json={"status": "failed", "progress": 0, "result": {"error": str(e)}})
                print(f"Worker: ошибка {task_id}: {e}")

if __name__ == "__main__":
    asyncio.run(worker_loop())
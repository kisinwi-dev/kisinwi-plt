import httpx
import asyncio
import uvicorn
from fastapi import FastAPI

from .api.routers import api_routers
from .core import training_model
from .logs import get_logger
from .service.tasker.tasker import Tasker_Service, TaskStatus
from .config import config_domain

logger = get_logger(__name__)

# Создание объекта fastapi
app = FastAPI(
    title="Trainer service",
    version="1.0.0",
    description="Занимается обучением DL моделей"
)

# Добавление роутеров
app.include_router(api_routers)

# Настройка конфига для запуска uvicorn
uv_conf = uvicorn.Config(
    app, 
    host="0.0.0.0", 
    port=6200, 
)

# Создание объекта сервера uvicorn
server = uvicorn.Server(uv_conf)

logger.debug("Обьект сервера создан")

async def to_work():
    """
        Запуск воркера с постоянным опросом сервиса задач
    """

    tasker_domen = config_domain.TASKER

    async with httpx.AsyncClient(timeout=30.0) as client:
        logger.info("Воркер начал работу")
        
        time_sleep = 1
        ts = Tasker_Service(client)

        while True:

            # Запрашиваем задачу
            task = await ts.get_next_task()

            if task is None:
                logger.debug(f"Повторный запрос через {time_sleep} сек.")
                await asyncio.sleep(time_sleep)
                continue

            task_id = task.task_id
            config = task.payload

            logger.info(f"Worker начинает работу над задачей: {task_id}")
            
            # Обновение статуса задачи (выполняется)
            await ts.update_status_task(task_id, TaskStatus.IN_PROGRESS, 0, "Задача принята воркером")
            
            try:
                # Процесс обучения
                training_model(task_id, config)

                # Завершение
                await ts.update_status_task(task_id, TaskStatus.COMPLETED, 100, "Задача выполнена")
                logger.info(f"Задача {task_id} завершена")
                
            except Exception as e:
                await ts.update_status_task(task_id, TaskStatus.FAILED, 0, "Задача завершена с ошибкой")
                logger.error(f"Ошибка {task_id}: {e}")

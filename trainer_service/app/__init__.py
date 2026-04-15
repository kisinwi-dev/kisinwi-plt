import httpx
import asyncio
import uvicorn
from fastapi import FastAPI

from .api.routers import api_routers
from .core import training_model
from .logs import get_logger
from .service.tasker import tasker_service, TaskStatus

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
    async with httpx.AsyncClient(timeout=30.0) as client:
        logger.info("Воркер начал работу")
        
        time_sleep = 1
        tasker_service.set_client(client)

        while True:

            # Запрашиваем задачу
            task = await tasker_service.get_next_task()

            if task is None:
                logger.debug(f"Повторный запрос через {time_sleep} сек.")
                await asyncio.sleep(time_sleep)
                continue

            task_id = task.task_id
            params = task.params

            logger.info(f"Worker начинает работу над задачей: {task_id}")
            
            # Обновение статуса задачи (выполняется)
            await tasker_service.update_status_task(0, status=TaskStatus.IN_PROGRESS, description="Задача принята воркером")
            
            try:
                # Процесс обучения
                await training_model(params)

                # Завершение
                await tasker_service.update_status_task(100, TaskStatus.COMPLETED, description="Задача выполнена")
                logger.info(f"Задача {task_id} выполнена")
                
            except Exception as e:
                await tasker_service.update_status_task(status=TaskStatus.FAILED, description="Задача завершена с ошибкой")
                logger.error(f"Ошибка {task_id}: {e}")

from .config import config_services

import os
import httpx
import asyncio
import uvicorn
from fastapi import FastAPI

from .api.routers import api_routers
from .core import training_model
from .logs import get_logger
from .service.tasker import tasker_service
from .service.ml_models import get_params, patch_model_status

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
    port=int(os.getenv("TRAINER_SERVICE_PORT", 6200)),
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
                await asyncio.sleep(time_sleep)
                continue

            task_id = task["id"]
            model_id = task["model_id"]
            
            try:
                # получаем параметры для запуска обучения
                params = await get_params(model_id)
                logger.info(f"Worker начинает работу над задачей: {task_id}")
            
                # Обновение статуса задачи (выполняется)
                await tasker_service.update_status_task(
                    status="running", 
                    percentages=0,
                    status_info="Задача принята воркером", 
                )

                await patch_model_status(model_id, "training")
                # Процесс обучения
                await training_model(params, model_id)

                # Завершение
                await tasker_service.update_status_task(
                    status="completed", 
                    percentages=100,
                    status_info="Задача выполнена", 
                )
                await patch_model_status(model_id, "completed")
                logger.info(f"Задача '{task_id}' по обучению модели '{model_id}' выполнена")

            except Exception as e:
                logger.error(f"Ошибка task_id='{task_id}' model_id='{model_id}': {e}", exc_info=True)
                await tasker_service.update_status_task(
                    status="failed",
                    status_info="Задача завершена с ошибкой",
                    error=f"Ошибка: {str(e)}"
                )
                try:
                    await patch_model_status(model_id, "draft")
                except Exception as status_error:
                    logger.error(f"Не удалось вернуть модель '{model_id}' в статус draft: {status_error}")

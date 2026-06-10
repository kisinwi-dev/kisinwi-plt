from .config import config_services

import os
import httpx
import asyncio
import torch
import uvicorn
from fastapi import FastAPI

from .api.routers import api_routers
from .core import training_model
from .core.exceptions import TaskCancelledError
from .logs import get_logger
from .service.tasker import tasker_service
from .service.ml_models import get_params, patch_model_status

logger = get_logger(__name__)

openapi_tags = [
    {
        "name": "Information",
        "description": "Справочная информация для подготовки конфигурации обучения: "
                       "пример конфига, доступные модели, оптимизаторы, планировщики, "
                       "метрики, аугментации, вычислительные устройства и health-check.",
    },
    {
        "name": "Validation",
        "description": "Проверка конфигурации обучения без запуска: соответствие схеме "
                       "и существование модели, оптимизатора, планировщика, метрик, "
                       "трансформаций, доступность устройства.",
    },
]

# Создание объекта fastapi
app = FastAPI(
    title="Trainer service",
    version="1.0.0",
    description="""
Сервис обучения DL моделей (классификация изображений).

Работает как воркер: опрашивает tasker, забирает задачи, обучает модель (PyTorch),
отправляет метрики в metrics и веса в ml_models. HTTP API предоставляет
справочную информацию для составления конфигурации обучения.
""",
    openapi_tags=openapi_tags,
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

        await recover_orphaned_tasks()

        while True:

            try:
                # Запрашиваем задачу
                task = await tasker_service.get_next_task()

                if task is None:
                    await asyncio.sleep(time_sleep)
                    continue

                task_id = task["id"]
                model_id = task["model_id"]
            except Exception as e:
                logger.error(f"Ошибка при получении задачи: {e}", exc_info=True)
                await asyncio.sleep(time_sleep)
                continue

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

            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"CUDA OOM task_id='{task_id}' model_id='{model_id}': {e}")
                torch.cuda.empty_cache()
                await report_task_failed(
                    task_id,
                    error="Не хватило памяти GPU: уменьшите batch_size или разрешение изображений"
                )
                try:
                    await patch_model_status(model_id, "draft")
                except Exception as status_error:
                    logger.error(f"Не удалось вернуть модель '{model_id}' в статус draft: {status_error}")

            except TaskCancelledError:
                # Статус задачи уже cancelled (его поставил tasker), не перетираем
                logger.info(f"Задача '{task_id}' отменена пользователем, модель '{model_id}' возвращена в draft")
                try:
                    await patch_model_status(model_id, "draft")
                except Exception as status_error:
                    logger.error(f"Не удалось вернуть модель '{model_id}' в статус draft: {status_error}")

            except Exception as e:
                logger.error(f"Ошибка task_id='{task_id}' model_id='{model_id}': {e}", exc_info=True)
                await report_task_failed(task_id, error=f"Ошибка: {str(e)}")
                try:
                    await patch_model_status(model_id, "draft")
                except Exception as status_error:
                    logger.error(f"Не удалось вернуть модель '{model_id}' в статус draft: {status_error}")


async def recover_orphaned_tasks():
    """
    Помечает осиротевшие задачи как failed.

    Trainer — единственный воркер, поэтому любая задача в статусе running
    на момент его старта была прервана падением/перезапуском.
    """
    try:
        tasks = await tasker_service.get_tasks(status="running")
    except Exception as e:
        logger.error(f"Не удалось проверить осиротевшие задачи: {e!r}")
        return

    for task in tasks:
        task_id = task["id"]
        model_id = task["model_id"]
        logger.warning(f"Найдена осиротевшая задача '{task_id}' (модель '{model_id}')")
        await report_task_failed(
            task_id,
            error="Обучение прервано перезапуском trainer"
        )
        try:
            await patch_model_status(model_id, "draft")
        except Exception as status_error:
            logger.error(f"Не удалось вернуть модель '{model_id}' в статус draft: {status_error}")


async def report_task_failed(
        task_id: str,
        error: str,
        attempts: int = 3,
        retry_delay: float = 5.0
):
    """Отправляет статус failed в сервис задач с повторами"""
    for attempt in range(1, attempts + 1):
        ok = await tasker_service.update_status_task(
            status="failed",
            status_info="Задача завершена с ошибкой",
            error=error,
            task_id=task_id
        )
        if ok:
            return
        logger.warning(
            f"Не удалось отправить статус failed задачи '{task_id}' "
            f"(попытка {attempt}/{attempts})"
        )
        if attempt < attempts:
            await asyncio.sleep(retry_delay)

    logger.critical(
        f"Статус failed задачи '{task_id}' не доставлен в сервис задач, "
        f"задача останется в статусе running"
    )

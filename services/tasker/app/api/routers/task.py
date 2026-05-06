from fastapi import APIRouter, Depends, Response, HTTPException, status

from app.api.schemas import TaskCreate, TaskUpdate
from app.core.train_models_tasks import TrainingTaskManager
from app.api.deps import get_training_task_manager
from app.core.utils import valid_uuid
from app.service.ml_models import models_is_exists
from app.logs import get_logger

logger = get_logger(__name__)

routers = APIRouter(
    prefix='/tasks',
    tags=['task']
)

@routers.post(
    "",
    summary="Создание задачи для обучения",
    responses={
        201: {"description": "Задача успешно создана"},
        400: {"description": "Полученные данные не валидны"},
        404: {"description": "Модель не найдена"},
        503: {"description": "Ошибка подключения к БД"}
    },
    status_code=status.HTTP_201_CREATED
)
async def create_task(
    task: TaskCreate, 
    manager: TrainingTaskManager = Depends(get_training_task_manager)
):
    try:
        # Валидация данных
        valid_uuid(task.model_id, on_error=True)
        if task.discussion_id:
            valid_uuid(task.discussion_id, on_error=True)
        
        # Проверка существования модели в сервисе метрик
        if not models_is_exists(task.model_id):
            logger.warning(f"МЛ модель '{task.model_id}' не существует")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Модель с ID {task.model_id} не найдена"
            )

        task_id = manager.create(
            name=task.task_name,
            model_id=task.model_id,
            discussion_id=task.discussion_id
        )
        return {"task_id": task_id}
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Невалидный UUID: {e}"
        )

@routers.get(
    "/pending",
    summary="Получение задач с статусом 'pending'"
)
async def get_pending(
    manager: TrainingTaskManager = Depends(get_training_task_manager)
):
    """Получаем список всех задач в статусе pending"""
    return manager.get_pending()

@routers.get(
    "/next",
    summary="Получение первой задачи в очереди"
)
async def next_task(
    manager: TrainingTaskManager = Depends(get_training_task_manager)
):
    """Воркер вызывает этот endpoint, чтобы получить следующую задачу."""
    tasks = manager.get_pending()
    if len(tasks) != 0:
        return tasks[0]
    else:
        return Response(status_code=204)


@routers.post(
    "/count",
    summary="Количество задач"
)
async def count_task(
    manager: TrainingTaskManager = Depends(get_training_task_manager)
):
    return manager.count_task()


@routers.delete(
    "/{task_id}",
    summary="Удаление задачи"
)
async def delete_task(
    task_id: str,
    manager: TrainingTaskManager = Depends(get_training_task_manager)
):
    return manager.delete(task_id)

@routers.get(
    "/{task_id}",
    summary="Получить информацию о задаче",
)
async def get_task_for_id(
    task_id: str,
    manager: TrainingTaskManager = Depends(get_training_task_manager)
):
    return manager.get_task(task_id)

@routers.post(
    "/{task_id}/status",
    summary="Обновить статус задачи"
)
async def update_task_status(
    task_id: str,
    update: TaskUpdate,
    manager: TrainingTaskManager = Depends(get_training_task_manager)
):
    manager.update_status(
        task_id=task_id,
        status=update.status,
        status_info=update.status_info,
        error=update.error
    )

    return True

@routers.post(
    "/{task_id}/agents-response",
    summary="Добавление id ответа агента к задаче"
)
async def add_agent_response(
    task_id: str,
    agent_respons: str,
    manager: TrainingTaskManager = Depends(get_training_task_manager)
):
    manager.add_agent_respons(
        task_id=task_id,
        agent_respons=agent_respons
    )

    return True
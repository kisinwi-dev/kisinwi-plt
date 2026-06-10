from fastapi import APIRouter, Depends, Response, HTTPException, status

from app.api.schemas import (
    TaskCreate, TaskUpdate, TaskStatistics, 
    TaskResponse, TasksResponse,
    TaskResponseMin
)
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
    "",
    summary="Получение всех задач",
    response_model=TasksResponse,
    responses={
        200: {"description": "Задачи успешно получены"},
        204: {"description": "Задачи не найдены"},
        503: {"description": "Ошибка подключения к БД"}
    },
    status_code=status.HTTP_201_CREATED
)
async def get_tasks(
    manager: TrainingTaskManager = Depends(get_training_task_manager)
):
    try:
        
        tasks = manager.get_tasks()

        if tasks is None:
            raise HTTPException(
                status_code=status.HTTP_204_NO_CONTENT,
                detail="Задачи не найдены"
            )

        return TasksResponse(
            tasks=[
                TaskResponse(**task)
                for task in tasks
            ]
        )
    except HTTPException:
        raise

@routers.get(
    "/next",
    summary="Получение первой задачи в очереди",
    response_model=TaskResponse,
    responses={
        200: {"description": "Задача успешно получена"},
        204: {"description": "Нет задач в очереди"},
        503: {"description": "Ошибка подключения к БД"}
    }
)
async def next_task(
    manager: TrainingTaskManager = Depends(get_training_task_manager)
):
    """Воркер вызывает этот endpoint, чтобы получить следующую задачу."""
    try:
        task = manager.get_next_task()
        
        if task is None:
            raise HTTPException(
                status_code=status.HTTP_204_NO_CONTENT,
                detail="Нет задач с статусом ожидания"
            )
        
        return TaskResponse(**task)

    except HTTPException:
        raise


@routers.get(
    "/count",
    summary="Количество задач",
    response_model=TaskStatistics,
    responses={
        200: {"description": "Количество задач успешно получено"},
        503: {"description": "Ошибка подключения к БД"}
    }
)
async def count_task(
    manager: TrainingTaskManager = Depends(get_training_task_manager)
):
    return TaskStatistics(
        count=manager.count_task()
    )


@routers.delete(
    "/{task_id}",
    summary="Удаление задачи",
    responses={
        204: {"description": "Задача успешно удалена"},
        404: {"description": "Задача не найдена"},
        400: {"description": "Невалидный UUID"},
        503: {"description": "Ошибка подключения к БД"}
    }
)
async def delete_task(
    task_id: str,
    manager: TrainingTaskManager = Depends(get_training_task_manager)
):
    try:
        valid_uuid(task_id, True)

        deleted = manager.delete(task_id)
        
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Задача с ID {task_id} не найдена"
            )
        
        return Response(
            status_code=status.HTTP_204_NO_CONTENT
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка удаления задачи {task_id}: {e}")
        raise

@routers.get(
    "/{task_id}",
    summary="Получить информацию о задаче",
    response_model=TaskResponse,
    responses={
        201: {"description": "Задача получена"},
        400: {"description": "Полученные данные не валидны"},
        404: {"description": "Задача не найдена"},
        503: {"description": "Ошибка подключения к БД"}
    },
)
async def get_task_for_id(
    task_id: str,
    manager: TrainingTaskManager = Depends(get_training_task_manager)
):
    valid_uuid(task_id, True)
    
    task = manager.get_task(task_id)
    
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Задача с ID {task_id} не найдена"
        )
    return TaskResponse(**task)
  

@routers.post(
    "/{task_id}/status",
    summary="Обновить статус задачи",
    responses={
        200: {"description": "Задача обновлена"},
        400: {"description": "Задача/статус не найдены"},
        503: {"description": "Ошибка подключения к БД"}
    },
)
async def update_task_status(
    task_id: str,
    update: TaskUpdate,
    manager: TrainingTaskManager = Depends(get_training_task_manager)
):
    try:
        manager.update_status(
            task_id=task_id,
            status=update.status,
            percentages=update.percentages,
            status_info=update.status_info,
            error=update.error
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@routers.post(
    "/{task_id}/agents-response",
    summary="Добавление id ответа агента к задаче",
    responses={
        200: {"description": "ID ответа агента успешно добавлен"},
        400: {"description": "Невалидный UUID или agent_response_id"},
        404: {"description": "Задача не найдена"},
        503: {"description": "Ошибка подключения к БД"}
    }
)
async def add_agent_response(
    task_id: str,
    agent_respons_id: str,
    manager: TrainingTaskManager = Depends(get_training_task_manager)
):
    try:
        # Валидация UUID
        valid_uuid(task_id, True)
        valid_uuid(agent_respons_id, True)
        result = manager.add_agent_respons(
            task_id=task_id,
            agent_respons_id=agent_respons_id
        )
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Задача с ID {task_id} не найдена"
            )
        
        return Response(
            status_code=status.HTTP_200_OK
        )
    except HTTPException:
        raise
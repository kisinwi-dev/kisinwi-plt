import asyncio
from typing import Optional
from fastapi import APIRouter, Depends

from app.api.schemas import TaskCreate, TaskUpdate, AddAgentResponse
from app.core.train_models_tasks import TrainingTaskManager
from app.api.deps import get_training_task_manager

routers = APIRouter()

@routers.post("/tasks")
async def create_task(
    task: TaskCreate, 
    manager: TrainingTaskManager = Depends(get_training_task_manager)
):
    task_id = manager.create(
        model_id=task.model_id,
        discussion_id=task.discussion_id
    )
    return {"task_id": task_id}


@routers.post("/tasks/next", response_model=Optional[dict])
async def next_task(
    manager: TrainingTaskManager = Depends(get_training_task_manager)
):
    """Воркер вызывает этот endpoint, чтобы получить следующую задачу."""
    return manager.get_pending()[0]

@routers.patch("/tasks/{task_id}/status")
async def update_task_status(
    update: TaskUpdate,
    manager: TrainingTaskManager = Depends(get_training_task_manager)
):
    return manager.update_status(
        task_id=update.task_id,
        status=update.status,
        error=update.error
    )

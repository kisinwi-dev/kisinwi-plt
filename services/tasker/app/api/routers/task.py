import asyncio
import uuid
from typing import Optional
from fastapi import APIRouter, Response, HTTPException

from app.api.schemas import TaskCreate, TaskUpdate
from app.core import tasks_db, task_queue

routers = APIRouter()

@routers.post("/tasks")
async def create_task(task: TaskCreate):
    task_id = str(uuid.uuid4())
    tasks_db[task_id] = {
        "status": "pending",
        "progress": 0,
        "result": None,
        "params": task.params
    }
    await task_queue.put(task_id)
    return {"task_id": task_id, "status": "pending"}

@routers.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    task = tasks_db.get(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    return {
        "task_id": task_id,
        "status": task["status"],
        "progress": task["progress"],
        "result": task["result"]
    }

# --- API для воркера (pull задачи и обновление статуса) ---
@routers.post("/tasks/next", response_model=Optional[dict])
async def next_task():
    """Воркер вызывает этот endpoint, чтобы получить следующую задачу."""
    try:
        # Неблокирующее получение (если очереди нет, вернём 204)
        task_id = await asyncio.wait_for(task_queue.get(), timeout=0.5)
    except asyncio.TimeoutError:
        return Response(status_code=204)
    
    task = tasks_db[task_id]
    # Помечаем как running (опционально, можно позже)
    task["status"] = "in_progress"
    return {"task_id": task_id, "params": task["params"]}

@routers.patch("/tasks/{task_id}/status")
async def update_task_status(task_id: str, update: TaskUpdate):
    """
    Обновление статуса задачи.
    
    - PATCH метод для частичного обновления
    - Все поля опциональные
    """
    try:
        task = tasks_db.get(task_id)
        if not task:
            raise HTTPException(404, "Task not found")
        
        if update.status is not None:
            task["status"] = update.status
        
        if update.progress is not None:
            if not 0 <= update.progress <= 100:
                raise HTTPException(422, "Progress must be between 0 and 100")
            task["progress"] = update.progress
        
        if update.result is not None:
            task["result"] = update.result
        
        if update.description is not None: 
            task["description"] = update.description
        
        print(task)
    except Exception as e:
        print("Error", e) 
        
    return {
        "status": "ok",
    }

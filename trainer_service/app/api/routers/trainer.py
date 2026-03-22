import multiprocessing
from fastapi import APIRouter, HTTPException, BackgroundTasks, Request

from app.logs import get_logger
from app.api.schemas import TaskConfig
from app.core import training_model

logger = get_logger(__name__)

router = APIRouter(tags=["training"])

MAX_CONCURRENT_TASKS = 2

@router.post("/train")
async def start_training(
    config: TaskConfig,
    background_tasks: BackgroundTasks,
    request: Request
):
    logger.info('Создание задачи обучения')
    task_id = config.task_id

    # проверяем, не запущена ли уже задача с таким id
    if task_id in request.app.state.tasks:
        raise HTTPException(status_code=400, detail=f"Задача (id:{task_id}) уже существует")

    # проверяем лимит активных задач
    if request.app.state.active_tasks >= MAX_CONCURRENT_TASKS:
        raise HTTPException(status_code=429, detail=f"Нельзя запускать больше {MAX_CONCURRENT_TASKS} задач")

    request.app.state.tasks[task_id] = {
        "status": "pending",
        "result": None,
        "error": None
    }

    # создаём процесс обучения, передаём tasker_url
    process = multiprocessing.Process(
        target=training_model,
        args=(
            config,
            request.app.state.tasks,
            request.app.state.tasker_url
        )
    )

    process.start()

    request.app.state.tasks[task_id].update({
        "status": "running",
        "pid": process.pid
    })
    request.app.state.active_tasks += 1

    background_tasks.add_task(monitor_process, task_id, process, request.app.state)
    return {"task_id": task_id, "status": "running"}

async def monitor_process(task_id: str, process: multiprocessing.Process, state):
    try:
        process.join()
    finally:
        state.active_tasks -= 1
        if task_id in state.tasks and state.tasks[task_id]["status"] == "running":
            if process.exitcode != 0:
                state.tasks[task_id]["status"] = "failed"
                state.tasks[task_id]["error"] = f"В процессе обучения получена ошибка: {process.exitcode}"
                logger.error(f"Ошибка в процессе {task_id}: код {process.exitcode}")
            else:
                state.tasks[task_id]["status"] = "completed"
                logger.info(f"Процесс {task_id} успешно завершён")

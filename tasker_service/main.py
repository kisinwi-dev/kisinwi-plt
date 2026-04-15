import uuid
import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Response
from pydantic import BaseModel
from typing import Dict, Optional, List

app = FastAPI()

# Хранилище задач в памяти
tasks_db: Dict[str, dict] = {}
# Очередь задач (task_id)
task_queue: asyncio.Queue = asyncio.Queue()

class TaskCreate(BaseModel):
    params: dict

class TaskUpdate(BaseModel):
    status: Optional[str] = None  # или TaskStatus enum
    progress: Optional[int] = None
    result: Optional[dict] = None
    description: Optional[str] = None

# --- API для клиента (UI) ---
@app.post("/tasks")
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

@app.get("/tasks/{task_id}")
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
@app.post("/tasks/next", response_model=Optional[dict])
async def next_task():
    """Воркер вызывает этот endpoint, чтобы получить следующую задачу."""
    try:
        # Неблокирующее получение (если очереди нет, вернём 204)
        task_id = await asyncio.wait_for(task_queue.get(), timeout=0.5)
    except asyncio.TimeoutError:
        return Response(status_code=204)
    
    task = tasks_db[task_id]
    # Помечаем как running (опционально, можно позже)
    task["status"] = "running"
    return {"task_id": task_id, "params": task["params"]}

@app.patch("/tasks/{task_id}/status")
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

# --- WebSocket для live-обновлений (без Redis, используем простой брокер) ---
# Храним активные WebSocket соединения по task_id
ws_connections: Dict[str, List[WebSocket]] = {}

async def notify_websocket(task_id: str, task_data: dict):
    if task_id in ws_connections:
        for ws in ws_connections[task_id][:]:  # копия
            try:
                await ws.send_json({
                    "task_id": task_id,
                    "status": task_data["status"],
                    "progress": task_data["progress"],
                    "result": task_data["result"]
                })
                if task_data["status"] in ("completed", "failed"):
                    # После завершения можно закрыть или оставить, решать вам
                    pass
            except:
                # удаляем нерабочее соединение
                ws_connections[task_id].remove(ws)

@app.websocket("/ws/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    await websocket.accept()
    if task_id not in ws_connections:
        ws_connections[task_id] = []
    ws_connections[task_id].append(websocket)
    try:
        # Отправить текущее состояние сразу
        task = tasks_db.get(task_id)
        if task:
            await websocket.send_json({
                "task_id": task_id,
                "status": task["status"],
                "progress": task["progress"],
                "result": task["result"]
            })
        # Ждём сообщения от клиента (чтобы понять, когда закрыть)
        while True:
            await websocket.receive_text()  # просто держим соединение
    except WebSocketDisconnect:
        ws_connections[task_id].remove(websocket)
        if not ws_connections[task_id]:
            del ws_connections[task_id]


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=6110,
        reload=True,
    )

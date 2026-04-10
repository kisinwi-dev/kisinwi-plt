import uvicorn
from fastapi import FastAPI
from app.api.routers import api_routers

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

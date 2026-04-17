import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers import routers
from app.core.health import check_bd_all

# Проверка состояния требуемых БД для правильной работы сервиса
check_bd_all()

# Создание `сервера`
app = FastAPI(
    title="Data Manipulation Service",
    version="0.1.0"
)

app.include_router(routers)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=6310,
        reload=True,
    )

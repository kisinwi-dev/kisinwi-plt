from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.routes import router
from app.redis_client import init_redis, close_redis

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_redis()
    print("Redis connected")
    
    yield
    
    await close_redis()
    print("Redis connection closed")

app = FastAPI(
    title="Redis Microservice",
    description="Сервис для работы с Redis",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(router)

@app.get("/health")
async def health():
    return {"status": "ok"}
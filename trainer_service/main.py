import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import multiprocessing

from app.api.routers import api_routers
from app.logs import get_logger

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.tasker_url = os.getenv("TASKER_SERVICE_URL", "http://tasker-service:8000")
    logger.info(f"Tasker Service URL: {app.state.tasker_url}")

    manager = multiprocessing.Manager()
    app.state.manager = manager
    app.state.tasks = manager.dict()
    app.state.active_tasks = 0
    logger.info("🚀 Trainer service started")
    
    yield

    manager.shutdown()
    logger.info("🔴 Trainer service shut down")

app = FastAPI(
    title="Gym Service",
    version="0.1.0",
    lifespan=lifespan
)

app.include_router(
    api_routers,
    prefix="/api"
)

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
        port=8000,
        reload=True,
    )
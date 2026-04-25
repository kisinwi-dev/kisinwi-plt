from fastapi import APIRouter
from .task import routers as routers_task

routers = APIRouter()

routers.include_router(routers_task)

@routers.get("/health")
async def healt_check():
    return {"status": "ok"}
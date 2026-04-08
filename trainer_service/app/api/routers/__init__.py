from fastapi import APIRouter
from .train_settings import routers as tr

api_routers = APIRouter()

api_routers.include_router(tr)

@api_routers.get("/health")
async def health_check():
    return {"status": "ok"}
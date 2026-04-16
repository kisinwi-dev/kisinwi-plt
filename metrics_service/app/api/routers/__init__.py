from fastapi import APIRouter

from .storage import api_routers

routers = APIRouter()

routers.include_router(api_routers)
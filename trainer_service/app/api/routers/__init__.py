from fastapi import APIRouter
from .trainer import router

api_routers = APIRouter()

api_routers.include_router(router)

__all__ = ['api_routers']
from fastapi import APIRouter
from . import datasets

api_routers = APIRouter()

api_routers.include_router(datasets.router)
# api_router.include_router(versions.router)
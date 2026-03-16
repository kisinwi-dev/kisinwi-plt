from fastapi import APIRouter
from . import datasets, versions

api_routers = APIRouter()

api_routers.include_router(datasets.router)
api_routers.include_router(versions.router)
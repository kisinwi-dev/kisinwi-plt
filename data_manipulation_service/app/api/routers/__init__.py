from fastapi import APIRouter
from . import datasets, versions, upload

api_routers = APIRouter()

api_routers.include_router(upload.router)
api_routers.include_router(datasets.router)
api_routers.include_router(versions.router)
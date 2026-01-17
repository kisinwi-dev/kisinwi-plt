from fastapi import APIRouter

from .dataset import router as dataset_router
from .versions import router as versions_router
from .classes import router as classes_router
from .files import router as files_router

routers = APIRouter()

routers.include_router(
    dataset_router,
    prefix="/datasets",
    tags=["datasets"]
)

routers.include_router(
    versions_router,
    prefix="/{dataset_name}/versions",
    tags=["versions"],
)

routers.include_router(
    classes_router,
    prefix="/{dataset_name}/{version_name}/classes",
    tags=["classes"]
)

routers.include_router(
    files_router,
    prefix="/{dataset_name}/{version_name}/{class_name}",
    tags=["files"]
)
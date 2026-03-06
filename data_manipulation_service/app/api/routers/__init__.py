from fastapi import APIRouter

from .test_api import router as test_api

routers = APIRouter()

routers.include_router(
    test_api,
    prefix="/tests"
)

from fastapi import APIRouter

from .storage import router as r_s

routers = APIRouter()

routers.include_router(r_s)
from fastapi import APIRouter

from .storage import router as r_s
from .agents import router as r_a

routers = APIRouter()

routers.include_router(r_s)
routers.include_router(r_a)
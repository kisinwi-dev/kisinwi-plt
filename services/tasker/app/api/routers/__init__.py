from fastapi import APIRouter
from .task import routers as r_t
from .info import routers as r_i

routers = APIRouter()

routers.include_router(r_t)
routers.include_router(r_i)
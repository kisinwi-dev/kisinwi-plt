from fastapi import APIRouter
from .info import routers as r_i
from .discussion import router as r_d
from .response import router as r_a

routers = APIRouter()

routers.include_router(r_i)
routers.include_router(r_d)
routers.include_router(r_a)

from fastapi import APIRouter
from .info import routers as r_i
from .enpoints import router as r_a

routers = APIRouter()

routers.include_router(r_i)
routers.include_router(r_a)

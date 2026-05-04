from fastapi import APIRouter
from .info import routers as r_i

routers = APIRouter()

routers.include_router(r_i)
from fastapi import APIRouter
from .info import routers as r_i
from .ml_models import routers as r_m

routers = APIRouter()

routers.include_router(r_i)
routers.include_router(r_m)
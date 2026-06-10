from fastapi import APIRouter
from .info import routers as r_i
from .ml_models import routers as r_m
from .versions import routers as r_v
from .files import routers as r_f

routers = APIRouter()

routers.include_router(r_i)
routers.include_router(r_m)
routers.include_router(r_v)
routers.include_router(r_f)
from fastapi import APIRouter
from .info import routers as r_i
from .discussions import router as r_d
from .responses import router as r_r
from .agent_system import router as r_as
from .tools import router as r_t

routers = APIRouter()

routers.include_router(r_i)
routers.include_router(r_d)
routers.include_router(r_t)
routers.include_router(r_r)
routers.include_router(r_as)


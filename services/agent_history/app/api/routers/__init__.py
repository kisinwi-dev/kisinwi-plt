from fastapi import APIRouter
from .info import routers as r_i
from .discussions import router as r_d
from .responses import router as r_r
from .agent_system import router as r_as
from .agent_tools import router as r_at

routers = APIRouter()

routers.include_router(r_i)
routers.include_router(r_d)
routers.include_router(r_at)
routers.include_router(r_r)
routers.include_router(r_as)


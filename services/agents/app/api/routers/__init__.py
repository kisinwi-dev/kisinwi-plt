from fastapi import APIRouter
from .analytic import routers as r_a
from .ml_engin import routers as r_ml
from .searcher import routers as r_s
from .researcher import routers as r_r
from .full_pipeline import routers as r_fp
from .info import routers as r_i

routers = APIRouter()
routers.include_router(r_a)
routers.include_router(r_r)
routers.include_router(r_s)
routers.include_router(r_ml)
routers.include_router(r_fp)
routers.include_router(r_i)

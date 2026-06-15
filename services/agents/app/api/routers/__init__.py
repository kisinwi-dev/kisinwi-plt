from fastapi import APIRouter
from .analytic import routers as r_a
from .ml_engin import routers as r_ml
from .searcher import routers as r_s
from .researcher import routers as r_r
from .full_pipeline import routers as r_fp
from .quick_pipeline import routers as r_qp
from .info import routers as r_i
from .settings import routers as r_set

routers = APIRouter()
routers.include_router(r_i)
routers.include_router(r_set)
routers.include_router(r_a)
routers.include_router(r_r)
routers.include_router(r_s)
routers.include_router(r_ml)
routers.include_router(r_fp)
routers.include_router(r_qp)


from fastapi import APIRouter
from .analytic import routers as r_analytic
from .ml_engin import routers as r_ml
from .full_pipeline import routers as r_fp
from .info import routers as r_i

routers = APIRouter()
routers.include_router(r_analytic)
routers.include_router(r_ml)
routers.include_router(r_fp)
routers.include_router(r_i)

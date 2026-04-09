from fastapi import APIRouter
from .analytic import routers as r_analytic
from .ml import routers as r_ml
from .full_pipeline import routers as r_fp

routers = APIRouter()
routers.include_router(r_analytic)
routers.include_router(r_ml)
routers.include_router(r_fp)

@routers.get("/health")
def health_status():
    return {"status": "Ok"}

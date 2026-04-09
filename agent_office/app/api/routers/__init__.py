from fastapi import APIRouter
from .start_analytic import routers as r_analytic

routers = APIRouter()
routers.include_router(r_analytic)

@routers.get("/health")
def health_status():
    return {"status": "Ok"}

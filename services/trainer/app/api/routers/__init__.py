from fastapi import APIRouter
from .train_settings import routers as tr
from .validate_config import routers as vc

api_routers = APIRouter()

api_routers.include_router(tr)
api_routers.include_router(vc)

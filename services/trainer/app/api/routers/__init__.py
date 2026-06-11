from fastapi import APIRouter
from .train_settings import router as tr
from .validate_config import router as vc

api_routers = APIRouter()

api_routers.include_router(tr)
api_routers.include_router(vc)

from fastapi import FastAPI
from app.api.routers import api_routers

app = FastAPI()

app.include_router(api_routers)
from fastapi import APIRouter

api_routers = APIRouter()

@api_routers.get("/health")
async def health_check():
    return {"status": "ok"}
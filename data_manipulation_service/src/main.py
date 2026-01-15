import uvicorn
from api.routers import routers
from fastapi import FastAPI

app = FastAPI(
    title="Data Manipulation Service",
    version="0.1.0"
)

app.include_router(
    routers,
    prefix="/api/v1"
)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# from app.core.exception.base import CoreException
from app.api import api_routers
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Data Manipulation Service",
    version="0.1.0"
)

app.include_router(
    api_routers,
    prefix="/api"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=6310,
        reload=True,
    )

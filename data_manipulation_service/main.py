import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.exception.base import CoreException
from app.api.routers import api_routers


app = FastAPI(
    title="Data Manipulation Service",
    version="0.1.0"
)

app.include_router(
    api_routers,
    prefix="/api"
)

@app.exception_handler(CoreException)
async def core_exception_handler(request: Request, exc: CoreException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.message,
            "detail": exc.detail
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )

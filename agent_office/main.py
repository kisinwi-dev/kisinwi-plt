import uvicorn
from fastapi import FastAPI
from app.api.routers import routers

app = FastAPI()
app.include_router(routers)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=6400, reload=True)
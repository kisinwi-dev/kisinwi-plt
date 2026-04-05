import uvicorn
import asyncio
from fastapi_app import app
from worker import worker_loop

async def main():
    server = uvicorn.Server(uvicorn.Config(app, host="0.0.0.0", port=6200))
    
    await asyncio.gather(
        server.serve(),
        worker_loop()
    )

if __name__ == "__main__":
    asyncio.run(main())
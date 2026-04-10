import asyncio
from server import server
from worker import worker_loop

async def main():
    # Запуск сервера uvicorn и асинхронной функции опроса
    await asyncio.gather(
        server.serve(),
        worker_loop()
    )

if __name__ == "__main__":
    asyncio.run(main())
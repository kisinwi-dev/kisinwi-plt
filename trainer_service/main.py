import asyncio
from app import server, to_work

async def main():
    # Запуск сервера uvicorn и асинхронной функции опроса
    await asyncio.gather(
        server.serve(),
        to_work()
    )

if __name__ == "__main__":
    asyncio.run(main())
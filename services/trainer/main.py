import asyncio
from app import server, to_work, config_services

async def main():
    # Проверка доступа к вспомогательным сервисам
    await config_services.check_services()

    # Запуск сервера uvicorn и асинхронной функции опроса
    await asyncio.gather(
        server.serve(),
        to_work()
    )

if __name__ == "__main__":
    asyncio.run(main())

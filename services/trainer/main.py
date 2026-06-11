import asyncio
from app import server, config_services
from app.core.worker import to_work

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

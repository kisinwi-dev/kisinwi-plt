import pymongo
from pymongo.errors import ConnectionFailure
from app.config import mongodb_config
from app.logs import get_logger

logger = get_logger(__name__)

def check_connection_status(
        url: str,
        bd_name: str
    ):
    try:
        logger.debug(f"Проверка URL: {url}")
        client = pymongo.MongoClient(
            url,
            connectTimeoutMS=5000,
            socketTimeoutMS=5000
        )
        client[bd_name].command("ping")
        logger.info(f"🟩 MongoDB[{bd_name}]: готов")
    except ConnectionFailure as e:
        logger.error(f"Не удалось установить соединение с MongoDB (DB:'{bd_name}'):\n{e}")
        raise e
    finally:
        if client:
            client.close()

def check_bd_all():
    """Проверка подключения к базам данных"""
    logger.info("Проверяем состояние подключения к базам данных...")
    check_connection_status(mongodb_config.URL_METRIC, mongodb_config.DATABASE_METRIC)
    logger.info("Проверка завершена")
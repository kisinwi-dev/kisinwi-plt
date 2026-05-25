import pymongo
from pymongo.errors import ConnectionFailure
from typing import Dict, List

from app.config import mongodb_config
from app.logs import get_logger

logger = get_logger(__name__)

def check_connection_status(
        url: str,
        bd_name: str
    ) -> bool:
    """Проверка подключения к БД"""
    try:
        logger.debug(f"Проверка URL: {url}")
        client = pymongo.MongoClient(
            url,
            timeoutMS=1000
        )
        client[bd_name].command("ping")
        logger.info(f"🟩 MongoDB[{bd_name}]: готов")
        return True
    except ConnectionFailure as e:
        logger.error(f"Не удалось установить соединение с MongoDB (DB:'{bd_name}'):\n{e}")
        return False
    finally:
        if client:
            client.close()

def check_health_all() -> Dict[str, bool]:
    """
    Проверка подключения к базам данных

    Returns:
        Возвращает словарь с информацией по состоянию сервиса
    """
    logger.info("Проверяем состояние подключения к базам данных...")
    

    bd = mongodb_config.DATABASE_METRIC+"_mongo"
    connection_status_bd = check_connection_status(
        mongodb_config.URL_METRIC, 
        mongodb_config.DATABASE_METRIC
    )

    check_info = {
        bd:connection_status_bd
    }

    logger.info("Проверка завершена")
    return check_info
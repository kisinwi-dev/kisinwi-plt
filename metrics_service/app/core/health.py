import pymongo
from app.config import mongodb_config
from app.logs import get_logger

logger = get_logger(__name__)

def check_connection_status(
        url: str,
        bd_name: str
    ):
    try:
        logger.debug(f" Проверка URL: {url}")
        client = pymongo.MongoClient(url)
        client[bd_name].command("ping")
        logger.info(f"🟩 MongoDB[{bd_name}]: готов")
    except Exception as e:
        logger.warning(f"⚠️ Сервис не может установить соединение с {bd_name}:{e}")

def check_bd_all():
    """Проверка подключения к базам данных"""
    logger.info("Проверяем состояние подключения к базам данных...")
    check_connection_status(mongodb_config.URL_CV_METRIC, mongodb_config.DATABASE_CV_METRIC)
    logger.info("Проверка завершена")
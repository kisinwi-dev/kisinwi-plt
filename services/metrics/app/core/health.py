import pymongo
from pymongo.errors import PyMongoError

from app.api.schemas.info import HealthResponse, HealthStatus
from app.config import mongodb_config
from app.logs import get_logger

logger = get_logger(__name__)

def check_connection_status(
        url: str,
        bd_name: str
    ) -> HealthStatus:
    """Проверка подключения к БД"""
    client = None
    try:
        logger.debug(f"Проверка URL: {url}")
        client = pymongo.MongoClient(
            url,
            timeoutMS=1000
        )
        client[bd_name].command("ping")
        logger.info(f"🟩 MongoDB[{bd_name}]: готов")
        return HealthStatus.HEALTHY
    except PyMongoError as e:
        logger.error(f"Не удалось установить соединение с MongoDB (DB:'{bd_name}'):\n{e}")
        return HealthStatus.UNHEALTHY
    finally:
        if client:
            client.close()

def check_health_all() -> HealthResponse:
    """
    Проверка подключения к базам данных

    Returns:
        Возвращает словарь с информацией по состоянию сервиса
    """
    status = HealthStatus.HEALTHY
    logger.info("Проверяем состояние подключения к базам данных...")

    services = {
        "bd": check_connection_status(
            mongodb_config.URL_METRIC, 
            mongodb_config.DATABASE_METRIC
        )
    }
    
    for service_val in services.values():
        if service_val != status:
          status = HealthStatus.DEGRADED
          break  
    
    logger.info("Проверка завершена")
    return HealthResponse(
        status=status,
        services=services
    )
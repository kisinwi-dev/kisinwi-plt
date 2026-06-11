import psycopg2
from psycopg2 import OperationalError

from app.api.schemas.info import HealthResponse, HealthStatus
from app.service.ml_models import check_health_ml_models
from app.config import postgresql_config
from app.logs import get_logger

logger = get_logger(__name__)


def check_connection_status(
    url: str,
    bd_name: str
) -> HealthStatus:
    """Проверка подключения к PostgreSQL"""
    conn = None
    try:
        logger.debug(f"Проверка подключения к PostgreSQL: {bd_name}")
        logger.debug(f"URL: {url}")
        
        # Подключение к PostgreSQL
        conn = psycopg2.connect(url)
        
        # Проверка подключения через простой запрос
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
        
        logger.info(f"🟩 PostgreSQL[{bd_name}]: готов")
        return HealthStatus.HEALTHY
        
    except OperationalError as e:
        logger.error(f"Не удалось установить соединение с PostgreSQL (DB:'{bd_name}'):\n{e}")
        return HealthStatus.UNHEALTHY
    except Exception as e:
        logger.error(f"Неожиданная ошибка при подключении к PostgreSQL (DB:'{bd_name}'):\n{e}")
        return HealthStatus.UNHEALTHY
    finally:
        if conn:
            conn.close()


def check_health_all() -> HealthResponse:
    """
    Проверка подключения к базам данных

    Returns:
        Возвращает информацию по состоянию сервиса
    """
    status = HealthStatus.HEALTHY
    logger.info("Проверяем состояние подключения к базам данных...")
    services = {
        "bd": check_connection_status(
            postgresql_config.URL,
            postgresql_config.DATABASE
        )
    }

    logger.info("Проверяем состояние подключения к сторонним сервисам...")
    services["ml models"] = check_health_ml_models()

    for service_val in services.values():
        if service_val != status:
          status = HealthStatus.DEGRADED
          break  
    
    logger.info("Проверка завершена")
    return HealthResponse(
        status=status,
        services=services
    )
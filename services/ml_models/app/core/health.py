import psycopg2
from psycopg2 import OperationalError
from typing import Dict, List

from app.config import postgresql_config
from app.logs import get_logger

logger = get_logger(__name__)


def check_connection_status(
    url: str,
    bd_name: str,
    bd_info: str
) -> Dict[str, str]:
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
        return {
            "bd_info": bd_info,
            "status": "healthy"
        }
        
    except OperationalError as e:
        logger.error(f"Не удалось установить соединение с PostgreSQL (DB:'{bd_name}'):\n{e}")
        return {
            "bd_info": bd_info,
            "status": "dead"
        }
    except Exception as e:
        logger.error(f"Неожиданная ошибка при подключении к PostgreSQL (DB:'{bd_name}'):\n{e}")
        return {
            "bd_info": bd_info,
            "status": "dead"
        }
    finally:
        if conn:
            conn.close()


def check_health_all() -> List[Dict[str, str]]:
    """
    Проверка подключения к базам данных

    Returns:
        Возвращает список словарей с информацией по состоянию сервисов
    """
    logger.info("Проверяем состояние подключения к базам данных...")
    check_info = []
    
    check_info.append(
        check_connection_status(
            postgresql_config.URL,
            postgresql_config.DATABASE,
            "База данных задач"
        )
    )
    
    logger.info("Проверка завершена")
    return check_info
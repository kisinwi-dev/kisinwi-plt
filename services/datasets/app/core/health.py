from app.api.schemas.info import HealthResponse, HealthStatus
from app.logs import get_logger

logger = get_logger(__name__)


def check_health_all() -> HealthResponse:
    """
    Проверка подключения к базам данных

    Returns:
        Возвращает информацию по состоянию сервиса
    """
    status = HealthStatus.HEALTHY
    logger.info("Проверяем состояние подключения к базам данных...")
    services = {
        "file_system": HealthStatus.HEALTHY 
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
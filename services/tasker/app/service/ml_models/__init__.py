import requests
from fastapi import HTTPException, status

from app.api.schemas.info import HealthStatus
from app.config import ml_models_config
from app.logs import get_logger

logger = get_logger(__name__)

def models_is_exists(model_id: str):
    try:
        response = requests.get(
            ml_models_config.URL + '/models/' + model_id,
            timeout=5
        )
        return response.status_code == 200
    except requests.exceptions.Timeout:
        logger.error(f"Таймаут при проверке модели {model_id}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Сервер моделей не отвечает"
        )
    except requests.exceptions.ConnectionError:
        logger.error(f"Ошибка соединения при проверке модели {model_id}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Не удалось подключиться к серверу моделей"
        )
    except Exception as e:
        logger.error(f"Ошибка проверки модели {model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка проверки существования модели"
        )
    
def check_healt_ml_models():
    """Проверка работоспособности сервиса ML-моделей"""
    try:
        response = requests.get(
            ml_models_config.URL + '/info/health',
            timeout=5
        )
        return response.json()["status"]
    except requests.exceptions.Timeout:
        logger.error(f"Таймаут при проверке работоспособности сервиса ML-моделей")
        return HealthStatus.UNHEALTHY
    except requests.exceptions.ConnectionError:
        logger.error(f"Ошибка соединения")
        return HealthStatus.UNHEALTHY
    except Exception as e:
        logger.error(f"Непредвиденная ошибка: {e}")
        return HealthStatus.UNHEALTHY
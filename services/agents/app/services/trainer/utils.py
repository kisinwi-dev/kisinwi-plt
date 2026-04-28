import requests
from functools import wraps

from app.config import config_url
from app.logs import get_logger

logger = get_logger(__name__)

def get_json(endpoint: str, params: dict | None = None,) -> dict:
    resp = requests.get(
        f"{config_url.TRAINER_URL}{endpoint}", 
        params=params
    )
    resp.raise_for_status()
    return resp.json()

def handle_errors(func):
    """Декоратор для обработки ошибок API запросов"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.Timeout:
            error_str = f"Timeout: {func.__name__} не ответил..."
            logger.error(error_str)
            return {"ERROR": error_str}
        except requests.exceptions.ConnectionError:
            error_str = f"Connection error: Не удалось подключиться к {config_url.TRAINER_URL}"
            logger.error(error_str)
            return {"ERROR": error_str}
        except requests.exceptions.HTTPError as e:
            error_str = f"HTTP error {e.response.status_code}: {e.response.text}"
            logger.error(error_str)
            return {"ERROR": error_str}
        except Exception as e:
            error_str = f"Ошибка в {func.__name__}: {str(e)}"
            logger.error(error_str)
            return {"ERROR": error_str}
    return wrapper
import requests
from requests import Response
from functools import wraps

from app.logs import get_logger

logger = get_logger(__name__)

def handle_errors(domain: str):
    """
    Декоратор для обработки ошибок API запросов
    
    Args:
        domain: домен к которому мы подключаемся
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except requests.exceptions.Timeout:
                error_str = f"Timeout: {func.__name__} не ответил..."
                logger.error(error_str)
                return {"ERROR": error_str}
            except requests.exceptions.ConnectionError:
                error_str = f"Connection error: Не удалось подключиться к {domain}"
                logger.error(error_str)
                return {"ERROR": error_str}
            except requests.exceptions.HTTPError as e:
                
                if isinstance(e.response, Response):
                    error_str = f"HTTP error {e.response.status_code}: {e.response.text}"
                else:
                    error_str = f"HTTP error: Неизвестная ошибка..."
                logger.error(error_str)
                return {"ERROR": error_str}
            except Exception as e:
                error_str = f"Ошибка в {func.__name__}: {str(e)}"
                logger.error(error_str)
                return {"ERROR": error_str}
        return wrapper
    return decorator

def get_json(url: str, params: dict | None = None) -> dict:
    resp = requests.get(
        url,
        params=params
    )
    resp.raise_for_status()
    return resp.json()
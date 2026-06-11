import json
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


def post_json(url: str, data: dict | None = None) -> dict:
    resp = requests.post(
        url,
        json=data
    )
    resp.raise_for_status()
    return resp.json()


def tool_response(domain: str):
    """
    Декоратор для инструментов агентов: ловит ошибки и ВСЕГДА возвращает
    результат в виде текста. CrewAI ожидает от инструментов строковый вывод,
    поэтому dict сериализуется в читаемый JSON (с кириллицей и отступами).

    Args:
        domain: домен, к которому подключается инструмент (для текста ошибки)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs) -> str:
            try:
                result = func(*args, **kwargs)
            except requests.exceptions.Timeout:
                error_str = f"ERROR: Timeout — {func.__name__} не ответил"
                logger.error(error_str)
                return error_str
            except requests.exceptions.ConnectionError:
                error_str = f"ERROR: не удалось подключиться к {domain}"
                logger.error(error_str)
                return error_str
            except requests.exceptions.HTTPError as e:
                if isinstance(e.response, Response):
                    error_str = f"ERROR: HTTP {e.response.status_code}: {e.response.text}"
                else:
                    error_str = "ERROR: неизвестная HTTP ошибка"
                logger.error(error_str)
                return error_str
            except Exception as e:
                error_str = f"ERROR: ошибка в {func.__name__}: {str(e)}"
                logger.error(error_str)
                return error_str

            if isinstance(result, str):
                return result
            return json.dumps(result, ensure_ascii=False, indent=2)
        return wrapper
    return decorator
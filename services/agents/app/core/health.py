import openai
from openai import OpenAI
from typing import List, Dict

from app.logs import get_logger
from app.config import config_base_llm
from app.services.tasker import tasker
from app.services.ml_models import ml_models_client
from app.services.metrics import health as metrics_health
from app.services.trainer import health as trainer_health
from app.services.data import health as datasets_healt

logger = get_logger(__name__)

def check_connection_llm():
        
    client = OpenAI(
        api_key=config_base_llm.OPENROUTER_API_KEY,
        base_url=config_base_llm.OPENAI_API_BASE,
        timeout=10.0
    )

    try:
        _ = client.chat.completions.create(
            model=config_base_llm.OPENAI_MODEL_NAME,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
            timeout=5.0
        )
        
        return {
            "status": "healthy",
            "details": "Готово к использованию",
            "model": config_base_llm.OPENAI_MODEL_NAME,
            "api_base": config_base_llm.OPENAI_API_BASE
        }
    
    except openai.APITimeoutError as e:
        logger.error(f"Таймаут подключения к LLM: {e}")
        return {
            "status": "dead",
            "details": f"Таймаут подключения: {str(e)[:100]}",
            "model": config_base_llm.OPENAI_MODEL_NAME,
            "api_base": config_base_llm.OPENAI_API_BASE
        }
    
    except openai.APIConnectionError as e:
        logger.error(f"Ошибка соединения с LLM: {e}")
        return {
            "status": "dead",
            "details": f"Ошибка соединения: {str(e)[:100]}",
            "model": config_base_llm.OPENAI_MODEL_NAME,
            "api_base": config_base_llm.OPENAI_API_BASE
        }
    
    except openai.AuthenticationError as e:
        logger.error(f"Ошибка аутентификации LLM: {e}")
        return {
            "status": "dead",
            "details": "Ошибка аутентификации. Проверьте API ключ",
            "model": config_base_llm.OPENAI_MODEL_NAME,
            "api_base": config_base_llm.OPENAI_API_BASE
        }
    
    except Exception as e:
        logger.error(f"Неизвестная ошибка при проверке LLM: {e}")
        return {
            "status": "dead",
            "details": f"Неизвестная ошибка: {str(e)[:100]}",
            "model": config_base_llm.OPENAI_MODEL_NAME,
            "api_base": config_base_llm.OPENAI_API_BASE
        }

def check_health_all() -> Dict[str, List]:
    """
    Проверка подключения к базам данных

    Returns:
        Возвращает список словарей с информацией по состоянию сервисов
    """
    logger.info("Проверяка состояния...")
    
    info = {}
    logger.info(" - подключение к LLM...")
    info["llm"] = {
        "base_llm": check_connection_llm()
    }

    logger.info(" - подключение к внутренним сервисам...")
    info["services"] = {
        "datasets": datasets_healt(),
        "ml_models": ml_models_client.health(),
        "tasker": tasker.health(),
        "trainer": trainer_health(),
        "metrics": metrics_health()
    }

    logger.info("✅ Проверка завершена")
    return info
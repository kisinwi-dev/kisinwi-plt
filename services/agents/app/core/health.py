import openai
from crewai import LLM
from typing import List, Dict

from app.logs import get_logger
from app.config import config_base_llm, config_url

logger = get_logger(__name__)

def check_connection_llm():
        
    client = LLM(
        model=config_base_llm.OPENAI_MODEL_NAME,
        base_url=config_base_llm.OPENAI_API_BASE,
        api_key=config_base_llm.OPENROUTER_API_KEY,
        temperature=0.7,
    )

    try:
        _ = client.call(
            "Напиши Ок и всё. Я проверяю, что ты работаешь"
        )
        
        return {
            "healthy": "true",
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
    info["services"] = config_url.check_services()

    logger.info("✅ Проверка завершена")
    return info
import yaml
from uuid import uuid4
from pathlib import Path
from functools import wraps
from typing import Callable

from app.services.agent_history import agent_history_client
from app.core.memory import agent_response_context
from app.logs import get_logger

logger = get_logger(__name__)

def track_agent(agent_role: str):
    """
    Декоратор для отслеживания запуска и завершения работы агента.

    Args:
        agent_role: роль агента

    Usage:
        @track_agent(
            agent_role="praxis_searcher"
        )
        def run_praxis_searcher(...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            response_id = str(uuid4())
            agent_response_context.set(response_id)

            agent_history_client.agent_start(
                response_id=response_id,
                agent_role=agent_role,
                text=f"Агент '{agent_role}' начал работу",
            )
            logger.info(f"🚀 Агент '{agent_role}' начал работу (response_id={response_id})")

            try:
                result = func(*args, **kwargs)

                agent_history_client.agent_succeed(
                    response_id=response_id,
                    agent_role=agent_role,
                    text=str(result) if result else f"Агент '{agent_role}' завершил работу",
                )
                logger.info(f"✅ Агент '{agent_role}' завершил работу")

                return result

            except Exception as e:
                error_msg = f"Агент '{agent_role}' завершился с ошибкой: {str(e)}"
                agent_history_client.error(error_msg)
                logger.error(f"❌ {error_msg}")
                raise

            finally:
                agent_response_context.clear()

        return wrapper
    return decorator

def get_agent_role_from_config(
    agent_key: str, 
    agent_path: Path
) -> str:
    """
    Загружает роль агента из YAML файла.

    Args:
        agent_key: Ключ агента в YAML файле
        agent_path: Path к файлу с реализацией crew == Path(__file__)

    Returns:
        Роль агента из YAML или None, если не найдено

    Пример структуры:
        app/core/crews/praxis_searcher/
        ├── praxis_searcher.py  # agent_path указывает сюда
        └── config/
            └── agent.yaml      # здесь лежит конфиг
    """
    try:
        config_file = agent_path.parent / "config" / "agent.yaml"

        if not config_file.exists():
            logger.warning(f"Файл конфигурации не найден: {config_file}")
            return agent_key

        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # поиск агента
        agent_config = config.get(agent_key)
        if agent_config and isinstance(agent_config, dict):
            role = agent_config.get('role')
            if role:
                role = role.strip()
                logger.debug(f"Загружена роль '{role}' для агента '{agent_key}' из {config_file}")
                return role

        logger.warning(f"Ключ '{agent_key}' или поле 'role' не найдены в {config_file}")
        return agent_key

    except yaml.YAMLError as e:
        logger.error(f"Ошибка парсинга YAML файла {config_file}: {e}")
        return agent_key
    except Exception as e:
        logger.error(f"Ошибка загрузки роли агента: {e}")
        return agent_key

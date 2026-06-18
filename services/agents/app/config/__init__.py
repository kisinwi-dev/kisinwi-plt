import os
import requests
from dotenv import load_dotenv
from dataclasses import dataclass

from app.api.schemas import HealthResponse, HealthStatus
from app.logs import get_logger

logger = get_logger(__name__)

load_dotenv()

@dataclass
class ConfigServices:
    DATASETS = {
        'name': 'datasets',
        'url': f"http://{os.getenv('DATASETS_DOMAIN', 'localhost:6500')}"
    }
    TASKER = {
        'name': "tasker",
        'url': f"http://{os.getenv('TASKER_DOMAIN', 'localhost:6110')}"
    }
    TRAINER = {
        'name': "trainer",
        'url': f"http://{os.getenv('TRAINER_DOMAIN', 'localhost:6200')}"
    }
    ML_MODELS = {
        'name': "ml_models",
        'url': f"http://{os.getenv('ML_MODELS_DOMAIN', 'localhost:6300')}"
    }
    METRICS = {
        'name': "metrics",
        'url': f"http://{os.getenv('METRICS_DOMAIN', 'localhost:6310')}"
    }
    AGENT_HISTORY = {
        'name': "agent_history",
        'url': f"http://{os.getenv('AGENT_HISTORY_DOMAIN', 'localhost:6410')}"
    }

    ALL_SERVICES = [
        DATASETS, TASKER, TRAINER, ML_MODELS,
        METRICS, AGENT_HISTORY
    ]

    def check_services(self) -> HealthResponse:
        """
        Проверка работоспособности сервиса
        
        Returns:
            Слоаварь с названием сервиса и статусом
        """
        status = HealthStatus.HEALTHY
        services = {}
        logger.info(f"Проверка доступа к сервисам:")
        for service in self.ALL_SERVICES:
            service_name = service.get("name", "ERROR")
            services[service_name] = self._health(service)
            if services[service_name] != HealthStatus.HEALTHY:
                status = HealthStatus.DEGRADED

        return HealthResponse(
            status=status,
            services=services
        )
    
    def _health(
        self, 
        service: dict
    ) -> HealthStatus:
        try:
            service_url = service.get("url", "ERROR")
            service_name = service.get("name", "ERROR")

            response = requests.get(
                f"{service_url}/info/health",
                timeout=30
            )        
            response.raise_for_status()
            logger.info(f" ✅ Сервис `{service_name}` доступен")
            return response.json()["status"]
        except requests.RequestException as e:
            logger.error(f" 🟥 Ошибка HTTP при обращении к сервису `{service_name}`, `{service_url}`\nError:{e}")
            return HealthStatus.UNHEALTHY
        except Exception as e:
            logger.error(f" 🟥 Ошибка при обращении к сервису `{service_name}`: {e}")
            return HealthStatus.UNHEALTHY

def _env_float(name: str, default: float) -> float:
    """Прочитать float из env с fallback на дефолт при пустом/битом значении."""
    try:
        return float(os.getenv(name) or default)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    """Прочитать int из env с fallback на дефолт при пустом/битом значении."""
    try:
        return int(os.getenv(name) or default)
    except (TypeError, ValueError):
        return default


@dataclass
class ConfigBaseLLM:
    OPENAI_MODEL_NAME = str(os.getenv("OPENAI_MODEL_NAME"))
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    # Некоторые модели (напр. gpt-5*) принимают только temperature=1.
    # Если модель не поддерживает кастомный temperature выставить в "false".
    LLM_TEMPERATURE_SUPPORTED = os.getenv("LLM_TEMPERATURE_SUPPORTED", "true").lower() != "false"
    # Таймаут одного LLM-запроса (сек). Без него litellm ждёт ответ бесконечно,
    # и подвисший запрос к провайдеру вешает весь шаг пайплайна.
    LLM_REQUEST_TIMEOUT = _env_float("LLM_REQUEST_TIMEOUT", 120)
    # Сколько раз litellm автоматически повторит запрос при таймауте/временной ошибке.
    LLM_NUM_RETRIES = _env_int("LLM_NUM_RETRIES", 2)
    # Жёсткий потолок на работу одного агента (сек). Сверху ограничивает чтобы зависший агент не висел вечно.
    AGENT_MAX_EXECUTION_TIME = _env_int("AGENT_MAX_EXECUTION_TIME", 600)
    # Сколько подряд сетевых ошибок при опросе статуса обучения терпим, прежде чем
    # сдаться. Единичный blip к tasker не должен ронять многочасовое обучение.
    TRAINING_POLL_MAX_CONSEC_ERRORS = _env_int("TRAINING_POLL_MAX_CONSEC_ERRORS", 5)

config_url = ConfigServices()
config_base_llm = ConfigBaseLLM()
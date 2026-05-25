import os
import requests
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Dict

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

    def check_services(self) -> Dict[str, bool]:
        """
        Проверка доступа к сторонним сервисам
        
        Returns:
            Слоаварь с названием сервиса и статусом
        """
        result = dict()
        for service in self.ALL_SERVICES:
            service_name = service.get("name", "ERROR")
            result[service_name] = self._health(service)

        return result
    
    def _health(
        self, 
        service: dict
    ) -> bool:
        try:
            service_url = service.get("url", "ERROR")
            service_name = service.get("name", "ERROR")

            response = requests.get(
                f"{service_url}/info/health",
                timeout=30
            )        
            response.raise_for_status()
            
            return True
        except requests.RequestException as e:
            logger.error(f"Ошибка HTTP при обращении к сервису `{service_name}`, `{service_url}`\nError:{e}")
            return False
        except Exception as e:
            logger.error(f"Ошибка при обращении к сервису `{service_name}`: {e}")
            return False

@dataclass
class ConfigBaseLLM:
    OPENAI_MODEL_NAME = str(os.getenv("OPENAI_MODEL_NAME"))
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

config_url = ConfigServices()
config_base_llm = ConfigBaseLLM()
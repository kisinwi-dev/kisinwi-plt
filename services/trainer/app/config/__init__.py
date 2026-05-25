import os
import requests
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Dict

from app.logs import get_logger

logger = get_logger(__name__)

load_dotenv()

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    logger.warning("HF_TOKEN не найден в .env")
else:
    logger.info(f"✅ HF_TOKEN загружен ({hf_token[:10]}...)")

@dataclass
class ConfigServices:
    TASKER = {
        'name': "tasker",
        'url': f"http://{os.getenv('TASKER_DOMAIN', 'localhost:6110')}"
    }
    ML_MODELS = {
        'name': "ml_models",
        'url': f"http://{os.getenv("ML_MODELS_DOMAIN", "localhost:6300")}"
    }
    METRICS = {
        'name': "metrics",
        'url': f"http://{os.getenv("METRICS_DOMAIN", "localhost:6310")}"
    }

    ALL_SERVICES = [
        TASKER, ML_MODELS, METRICS
    ]

    def check_services(self) -> Dict[str, bool]:
        """
        Проверка доступа к сторонним сервисам
        
        Returns:
            Слоаварь с названием сервиса и статусом
        """
        result = dict()
        logger.info(f"Проверка доступа к сервисам:")
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
            logger.info(f" ✅ Сервис `{service_name}` доступен")
            return True
        except requests.RequestException as e:
            logger.error(f" 🟥 Ошибка HTTP при обращении к сервису `{service_name}`, `{service_url}`\nError:{e}")
            return False
        except Exception as e:
            logger.error(f" 🟥 Ошибка при обращении к сервису `{service_name}`: {e}")
            return False

config_services = ConfigServices()
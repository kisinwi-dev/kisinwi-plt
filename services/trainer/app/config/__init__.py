import os
import requests
from dotenv import load_dotenv
from dataclasses import dataclass

from app.api.schemas.health import HealthStatus, HealthResponse
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

config_services = ConfigServices()
import os
import asyncio
import httpx
from dotenv import load_dotenv
from dataclasses import dataclass

from app.api.schemas.health import HealthStatus, HealthResponse
from app.logs import get_logger

logger = get_logger(__name__)

load_dotenv()

if not os.getenv("HF_TOKEN"):
    logger.warning("HF_TOKEN не найден в .env")
else:
    logger.info("✅ HF_TOKEN загружен")

HEALTH_TIMEOUT = 5.0

@dataclass
class ConfigServices:
    TASKER = {
        'name': "tasker",
        'url': f"http://{os.getenv('TASKER_DOMAIN', 'localhost:6110')}"
    }
    ML_MODELS = {
        'name': "ml_models",
        'url': f"http://{os.getenv('ML_MODELS_DOMAIN', 'localhost:6300')}"
    }
    METRICS = {
        'name': "metrics",
        'url': f"http://{os.getenv('METRICS_DOMAIN', 'localhost:6310')}"
    }

    ALL_SERVICES = [
        TASKER, ML_MODELS, METRICS
    ]

    async def check_services(self) -> HealthResponse:
        """
        Проверка работоспособности вспомогательных сервисов

        Returns:
            Статус сервиса и статусы вспомогательных сервисов
        """
        status = HealthStatus.HEALTHY
        logger.info("Проверка доступа к сервисам:")

        async with httpx.AsyncClient(timeout=HEALTH_TIMEOUT) as client:
            results = await asyncio.gather(
                *[self._health(client, service) for service in self.ALL_SERVICES]
            )

        services = {}
        for service, service_status in zip(self.ALL_SERVICES, results):
            services[service['name']] = service_status
            if service_status != HealthStatus.HEALTHY:
                status = HealthStatus.DEGRADED

        return HealthResponse(
            status=status,
            services=services
        )

    async def _health(
        self,
        client: httpx.AsyncClient,
        service: dict
    ) -> HealthStatus:
        service_url = service.get("url", "ERROR")
        service_name = service.get("name", "ERROR")
        try:
            response = await client.get(f"{service_url}/info/health")
            response.raise_for_status()
            logger.info(f" ✅ Сервис `{service_name}` доступен")
            return response.json()["status"]
        except httpx.HTTPError as e:
            logger.error(f" 🟥 Ошибка HTTP при обращении к сервису `{service_name}`, `{service_url}`\nError:{e}")
            return HealthStatus.UNHEALTHY
        except Exception as e:
            logger.error(f" 🟥 Ошибка при обращении к сервису `{service_name}`: {e}")
            return HealthStatus.UNHEALTHY

config_services = ConfigServices()

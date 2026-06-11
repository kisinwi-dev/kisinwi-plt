from app.core.agent import AgentsResponseManager
from app.core.model import CVMetricManager
from app.core.stream import MetricStreamBroker
from app.config import mongodb_config

cv_training_metric_manager = CVMetricManager(
    url=mongodb_config.URL_METRIC,
    database_name=mongodb_config.DATABASE_METRIC,
    collection_name=mongodb_config.COLLECTION_TRAINING_CV
)

agents_metric_manager = AgentsResponseManager(
    url=mongodb_config.URL_METRIC,
    database_name=mongodb_config.DATABASE_METRIC,
    collection_name=mongodb_config.COLLECTION_AGENT_RESPONSE
)

def get_cv_training_metrics_manager() -> CVMetricManager:
    """Менеджер метрик обучения; подключается один раз в lifespan приложения"""
    return cv_training_metric_manager

def get_agent_metrics_manager() -> AgentsResponseManager:
    """Менеджер метрик агентов; подключается один раз в lifespan приложения"""
    return agents_metric_manager

metric_stream_broker = MetricStreamBroker()

def get_metric_stream_broker() -> MetricStreamBroker:
    """Брокер SSE-уведомлений об обновлении метрик; один на процесс"""
    return metric_stream_broker

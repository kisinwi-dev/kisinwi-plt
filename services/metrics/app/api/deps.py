from app.core.agent_response import AgentsResponseManager
from app.core.training_cv import CVMetricManager
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

async def get_cv_training_metrics_manager():
    try:
        cv_training_metric_manager.connect()
        yield cv_training_metric_manager
    finally:
        cv_training_metric_manager.disconnect()

async def get_agent_metrics_manager():
    try:
        agents_metric_manager.connect()
        yield agents_metric_manager
    finally:
        agents_metric_manager.disconnect()
from app.core.training_cv import CVMetricManager
from app.config import mongodb_config

manager = CVMetricManager(
    url=mongodb_config.URL_METRIC,
    database_name=mongodb_config.DATABASE_METRIC,
    collection_name=mongodb_config.COLLECTION_TRAINING_CV
)

async def get_metrics_manager():
    try:
        manager.connect()
        yield manager
    finally:
        manager.disconnect()
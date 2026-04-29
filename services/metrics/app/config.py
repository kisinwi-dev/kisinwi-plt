import os
from dataclasses import dataclass

@dataclass
class MongoDBConfig:
    HOST = os.getenv('MONGO_HOST', 'localhost')
    PORT = os.getenv('MONGO_PORT', '27017')
    USERNAME = os.getenv("MONGO_APP_USERNAME", "metrics_service")
    PASSWORD = os.getenv("MONGO_APP_PASSWORD", "060720")
    DATABASE_METRIC=os.getenv("MONGO_METRIC_DATABASE", "ml_metrics")
    COLLECTION_CV=os.getenv("MONGO_COLLECTION_CV", "task_cv")

    @property
    def URL_METRIC(self) -> str:
        return f"mongodb://{self.USERNAME}:{self.PASSWORD}@{self.HOST}:{self.PORT}/{self.DATABASE_METRIC}"


mongodb_config = MongoDBConfig()
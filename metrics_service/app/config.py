import os
from dataclasses import dataclass

@dataclass
class MongoDBConfig:
    HOST = os.getenv('MONGO_HOST', 'localhost:27017')
    PORT = os.getenv('MONGO_PORT', 'localhost:27017')
    USERNAME = os.getenv("MONGO_APP_USERNAME", "root")
    PASSWORD = os.getenv("MONGO_APP_PASSWORD", "123456")
    DATABASE_METRIC=os.getenv("MONGO_METRIC_DATABASE", "ml_metrics")
    COLLECTION_CV=os.getenv("MONGO_COLLECTION_CV", "task_cv")

    @property
    def URL_METRIC(self) -> str:
        return f"mongodb://{self.USERNAME}:{self.PASSWORD}@{self.HOST}:{self.PORT}/{self.DATABASE_METRIC}"


mongodb_config = MongoDBConfig()
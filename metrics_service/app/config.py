import os
from dataclasses import dataclass

@dataclass
class MongoDBConfig:
    HOST = os.getenv('MONGO_HOST', 'localhost:27017')
    PORT = os.getenv('MONGO_PORT', 'localhost:27017')
    USERNAME = os.getenv("MONGO_APP_USERNAME", "root")
    PASSWORD = os.getenv("MONGO_APP_PASSWORD", "123456")
    DATABASE_CV_METRIC=os.getenv("MONGO_CV_METRIC_DATABASE", "ml_metrics")

    @property
    def URL_CV_METRIC(self) -> str:
        return f"mongodb://{self.USERNAME}:{self.PASSWORD}@{self.HOST}:{self.PORT}/{self.DATABASE_CV_METRIC}"


mongodb_config = MongoDBConfig()
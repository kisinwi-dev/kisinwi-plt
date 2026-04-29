import os
from dataclasses import dataclass

@dataclass
class MongoDBConfig:
    HOST = os.getenv('MONGO_HOST', 'localhost')
    PORT = os.getenv('MONGO_PORT', '27017')
    USERNAME = os.getenv("MONGO_APP_USERNAME", "metrics_service")
    PASSWORD = os.getenv("MONGO_APP_PASSWORD", "060720")
    DATABASE_METRIC=os.getenv("MONGO_METRIC_DATABASE", "metrics")
    COLLECTION_TRAINING_CV=os.getenv("MONGO_COLLECTION_CV", "training_cv")
    COLLECTION_AGENT_RESPONSE=os.getenv("MONGO_COLLECTION_AGENT", "agent_response")

    @property
    def URL_METRIC(self) -> str:
        return f"mongodb://{self.USERNAME}:{self.PASSWORD}@{self.HOST}:{self.PORT}/{self.DATABASE_METRIC}"


mongodb_config = MongoDBConfig()
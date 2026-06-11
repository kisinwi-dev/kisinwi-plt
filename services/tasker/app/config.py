import os
from dataclasses import dataclass

@dataclass
class PostgreSQLConfig:
    HOST = os.getenv('POSTGRES_HOST', 'localhost')
    PORT = os.getenv('POSTGRES_PORT', '6115')
    USERNAME = os.getenv("POSTGRES_APP_USERNAME", "tasker_service")
    PASSWORD = os.getenv("POSTGRES_APP_PASSWORD")
    DATABASE = os.getenv("POSTGRES_DB", "task_service_db")

    @property
    def URL(self) -> str:
        if not self.PASSWORD:
            raise RuntimeError(
                "Не задана переменная окружения POSTGRES_APP_PASSWORD "
                "(локально: значение TASKER_POSTGRES_PASSWORD из корневого .env)"
            )
        return f"postgresql://{self.USERNAME}:{self.PASSWORD}@{self.HOST}:{self.PORT}/{self.DATABASE}"

@dataclass
class MLModelsServiceConfig:
    HOST = os.getenv('ML_MODELS_HOST', 'localhost')
    PORT = os.getenv('ML_MODELS_PORT', '6300')

    @property
    def URL(self) -> str:
        return f"http://{self.HOST}:{self.PORT}"

ml_models_config = MLModelsServiceConfig()
postgresql_config = PostgreSQLConfig()
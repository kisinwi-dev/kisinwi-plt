import os
from dataclasses import dataclass

@dataclass
class PostgreSQLConfig:
    HOST = os.getenv('POSTGRES_HOST', 'localhost')
    PORT = os.getenv('POSTGRES_PORT', '6305')
    USERNAME = os.getenv("POSTGRES_APP_USERNAME", "ml_models_service")
    PASSWORD = os.getenv("POSTGRES_APP_PASSWORD", "060720")
    DATABASE = os.getenv("POSTGRES_DB", "ml_models_service_db")

    @property
    def URL(self) -> str:
        return f"postgresql://{self.USERNAME}:{self.PASSWORD}@{self.HOST}:{self.PORT}/{self.DATABASE}"


postgresql_config = PostgreSQLConfig()
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

from app.logs import get_logger

logger = get_logger(__name__)

class ManagerBase:
    def __init__(
        self,
        url: str,
        database_name: str,
        collection_name: str
    ) -> None:
        self.url = url
        self.database_name = database_name
        self.collection_name = collection_name
        self.client: MongoClient
        self.collection: Collection

    def connect(self):
        """Подключение к MongoDB"""
        try:
            # Заходим в коллекцию
            self.client = MongoClient(self.url)
            self.db = self.client[self.database_name]
            self.collection = self.db[self.collection_name]

            logger.debug(f"🟩 {self.database_name} подключена")
        except PyMongoError as e:
            logger.error(f"😡 Не удалось подключиться к {self.database_name}/{self.collection_name} : {e}")
            raise
    
    def disconnect(self):
        """Отключение от MongoDB"""
        if self.client:
            self.client.close()
            logger.debug(f"⚠️ Соединение с {self.database_name}/{self.collection_name} закрыто")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self):
        self.disconnect()
        return False

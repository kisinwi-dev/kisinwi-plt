import psycopg2
from psycopg2 import OperationalError

from app.logs import get_logger

logger = get_logger(__name__)


class PostgresManager:
    """Базовый класс для работы с PostgreSQL (минимальная версия)"""
    
    def __init__(self, url: str):
        self.url = url
    
    def connect(self):
        """Подключение к БД"""
        try:
            self.conn = psycopg2.connect(self.url)
            self.cursor = self.conn.cursor()
            logger.info("✅ Подключено к PostgreSQL")
        except OperationalError as e:
            logger.error(f"Ошибка подключения: {e}")
            raise
    
    def disconnect(self):
        """Отключение от БД"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
    
    def execute(
            self, 
            query: str, 
            params: tuple | None = None
    ):
        """Выполнить запрос"""
        try:
            self.cursor.execute(query, params)
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Ошибка: {e}")
            raise
    
    def fetch_one(
            self, 
            query: str, 
            params: tuple | None = None
    ):
        """Выполнить запрос и вернуть одну строку"""
        self.execute(query, params)
        return self.cursor.fetchone()
    
    def fetch_all(
            self, 
            query: str, 
            params: tuple | None = None
    ):
        """Выполнить запрос и вернуть все строки"""
        self.execute(query, params)
        return self.cursor.fetchall()
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False
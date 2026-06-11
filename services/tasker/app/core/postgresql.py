from contextlib import contextmanager

import psycopg2
from psycopg2 import OperationalError
from psycopg2.extras import RealDictCursor

from app.logs import get_logger

logger = get_logger(__name__)


class PostgresSession:
    """Одно соединение + одна транзакция. Строки возвращаются как dict (RealDictCursor)"""

    def __init__(self, conn):
        self.conn = conn
        self.cursor = conn.cursor(cursor_factory=RealDictCursor)

    def execute(
            self,
            query: str,
            params: tuple | None = None
    ):
        """Выполнить запрос"""
        try:
            self.cursor.execute(query, params)
        except Exception as e:
            logger.error(f"Ошибка выполнение запроса:\nQuery:\n{query}\n\nParams:\n{params}\n\nError:\n{e}")
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


class PostgresManager:
    """Фабрика сессий PostgreSQL.

    Состояние соединения не хранится на инстансе — каждая сессия получает
    своё соединение, поэтому один менеджер безопасно делить между запросами.
    """

    def __init__(self, url: str):
        self.url = url

    @contextmanager
    def session(self):
        """Новое соединение и транзакция: commit при успехе, rollback при ошибке"""
        try:
            conn = psycopg2.connect(self.url)
        except OperationalError as e:
            logger.error(f"Ошибка подключения: {e}")
            raise

        try:
            yield PostgresSession(conn)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

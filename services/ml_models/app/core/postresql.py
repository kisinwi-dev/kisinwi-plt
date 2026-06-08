import threading

import psycopg2
from psycopg2 import OperationalError
from psycopg2.pool import ThreadedConnectionPool

from app.logs import get_logger

logger = get_logger(__name__)


class PostgresManager:
    """
    Базовый класс для работы с PostgreSQL через пул соединений.

    Пул общий для всех экземпляров с одним URL (class-level). Состояние
    соединения хранится в thread-local, поэтому конкурентные запросы (sync-роуты
    FastAPI бегут в threadpool) не делят один курсор. Интерфейс прежний:
    `with self.db as db: db.fetch_all(...)`.
    """

    _pool: ThreadedConnectionPool | None = None
    _pool_lock = threading.Lock()

    def __init__(self, url: str, minconn: int = 1, maxconn: int = 10):
        self.url = url
        self._local = threading.local()

        if PostgresManager._pool is None:
            with PostgresManager._pool_lock:
                if PostgresManager._pool is None:
                    try:
                        PostgresManager._pool = ThreadedConnectionPool(
                            minconn, maxconn, dsn=url
                        )
                        logger.info("✅ Создан пул соединений PostgreSQL")
                    except OperationalError as e:
                        logger.error(f"Ошибка создания пула соединений: {e}")
                        raise

    def execute(
            self,
            query: str,
            params: tuple | None = None
    ):
        """Выполнить запрос"""
        try:
            self._local.cursor.execute(query, params)
            self._local.conn.commit()
        except Exception as e:
            self._local.conn.rollback()
            logger.error(f"Ошибка выполнение запроса:\nQuery:\n{query}\n\nParams:\n{params}\n\nError:\n{e}")
            raise

    def fetch_one(
            self,
            query: str,
            params: tuple | None = None
    ):
        """Выполнить запрос и вернуть одну строку"""
        self.execute(query, params)
        return self._local.cursor.fetchone()

    def fetch_all(
            self,
            query: str,
            params: tuple | None = None
    ):
        """Выполнить запрос и вернуть все строки"""
        self.execute(query, params)
        return self._local.cursor.fetchall()

    def __enter__(self):
        conn = PostgresManager._pool.getconn()
        self._local.conn = conn
        self._local.cursor = conn.cursor()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        cursor = getattr(self._local, "cursor", None)
        conn = getattr(self._local, "conn", None)
        if cursor is not None:
            cursor.close()
        if conn is not None:
            PostgresManager._pool.putconn(conn)
        self._local.cursor = None
        self._local.conn = None
        return False

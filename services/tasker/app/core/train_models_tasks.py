import json
from typing import List, Dict, Any

from .postgresql import PostgresManager
from app.config import postgresql_config
from app.logs import get_logger

logger = get_logger(__name__)

FINAL_STATUSES = ('completed', 'failed', 'cancelled')

# Общий список полей задачи (одинаковый порядок ключей во всех выборках)
TASK_FIELDS = """
    t.id,
    t.name,
    t.model_id,
    t.discussion_id,
    t.agent_respons_ids,
    t.status_id,
    s.status AS status,
    s.description AS status_description,
    t.percentages,
    t.status_info,
    t.error_message,
    t.created_at,
    t.started_at,
    t.updated_at,
    t.completed_at
"""


class TrainingTaskManager:
    def __init__(self):
        self.db = PostgresManager(postgresql_config.URL)
        self._table = "train_models_tasks"
        self._status_table = "task_statuses"

    def count_task(self) -> int:
        """Выводит количество имеющихся задач"""
        query = f"""
            SELECT COUNT(id) AS count
            FROM {self._table}
        """
        with self.db.session() as db:
            result = db.fetch_one(query)
            return result["count"] if result else 0

    def create(
            self,
            model_id: str,
            name: str,
            discussion_id: str | None = None
    ) -> str:
        """Создание задачи"""
        query = f"""
            INSERT INTO {self._table} (model_id, name, discussion_id)
            VALUES (%s, %s, %s)
            RETURNING id
        """
        with self.db.session() as db:
            result = db.fetch_one(query, (model_id, name, discussion_id))

            if not result:
                raise RuntimeError(f"Ошибка создания задачи для обучения модели({model_id})")

            task_id = result["id"]
            logger.info(f"✅ Создана задача {task_id} для модели {model_id}")
            return task_id

    def delete(
        self,
        task_id: str,
    ) -> bool:
        """Удаление задачи"""
        query = f"""
            DELETE FROM {self._table}
            WHERE id = %s
            RETURNING id
        """
        with self.db.session() as db:
            row = db.fetch_one(query, (task_id,))
            return row is not None

    def update_status(
        self,
        task_id: str,
        status: str,
        status_info: str,
        percentages: int | None = None,
        error: str | None = None
    ):
        """Обновить статус"""
        with self.db.session() as db:

            status_row = db.fetch_one(
                f"SELECT id FROM {self._status_table} WHERE status = %s",
                (status,)
            )

            if not status_row:
                raise ValueError(f"Неизвестный статус: {status}")

            # Условный UPDATE: задача в финальном статусе не обновляется
            # (иначе progress-апдейты воркера перетирают cancelled/completed/failed).
            # updated_at / started_at / completed_at проставляет триггер БД.
            query = f"""
                UPDATE {self._table}
                SET status_id = %s,
                    status_info = %s,
                    error_message = COALESCE(%s, error_message),
                    percentages = COALESCE(%s, percentages)
                WHERE id = %s
                  AND status_id NOT IN (
                      SELECT id FROM {self._status_table} WHERE status IN %s
                  )
                RETURNING id
            """
            result = db.fetch_one(
                query,
                (status_row["id"], status_info, error, percentages, task_id, FINAL_STATUSES)
            )

            if result is None:
                current = db.fetch_one(
                    f"""
                    SELECT s.status FROM {self._table} t
                    LEFT JOIN {self._status_table} s ON t.status_id = s.id
                    WHERE t.id = %s
                    """,
                    (task_id,)
                )
                if current is None:
                    raise ValueError(f"Задача с ID {task_id} не найдена")
                raise ValueError(
                    f"Задача в финальном статусе '{current['status']}' не может быть обновлена"
                )

            logger.info(f"Статус задачи {task_id} обновлён на '{status}'")

    def get_task(
            self,
            task_id: str
    ) -> Dict | None:
        """Получить информацию о задаче"""
        query = f"""
            SELECT {TASK_FIELDS}
            FROM {self._table} t
            LEFT JOIN {self._status_table} s ON t.status_id = s.id
            WHERE t.id = %s
        """
        with self.db.session() as db:
            row = db.fetch_one(query, (task_id,))
            return dict(row) if row is not None else None

    def get_tasks(
            self,
            status: str | None = None,
            model_id: str | None = None
    ) -> List[Dict]:
        """Получить информацию о задачах (с опциональными фильтрами)"""
        conditions = []
        params: list[Any] = []
        if status:
            conditions.append("s.status = %s")
            params.append(status)
        if model_id:
            conditions.append("t.model_id = %s")
            params.append(model_id)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"""
            SELECT {TASK_FIELDS}
            FROM {self._table} t
            LEFT JOIN {self._status_table} s ON t.status_id = s.id
            {where}
            ORDER BY t.created_at ASC
        """
        with self.db.session() as db:
            rows = db.fetch_all(query, tuple(params) if params else None)
            return [dict(row) for row in rows]

    def get_status_values(self) -> List[Dict[str, Any]]:
        """Получить все возможные значения статуса"""
        query = f"""
            SELECT id, status, description
            FROM {self._status_table}
        """
        with self.db.session() as db:
            rows = db.fetch_all(query)
            return [dict(row) for row in rows]

    def get_next_task(self) -> Dict[str, Any] | None:
        """Атомарно забрать первую задачу из очереди.

        Задача сразу переводится в running (FOR UPDATE SKIP LOCKED),
        поэтому два воркера не получат одну и ту же задачу.
        """
        claim_query = f"""
            UPDATE {self._table}
            SET status_id = (SELECT id FROM {self._status_table} WHERE status = 'running'),
                status_info = 'Задача принята воркером'
            WHERE id = (
                SELECT t.id
                FROM {self._table} t
                JOIN {self._status_table} s ON t.status_id = s.id
                WHERE s.status = 'waiting'
                ORDER BY t.created_at ASC
                LIMIT 1
                FOR UPDATE SKIP LOCKED
            )
            RETURNING id
        """
        with self.db.session() as db:
            claimed = db.fetch_one(claim_query)

            if claimed is None:
                return None

            task = db.fetch_one(
                f"""
                SELECT {TASK_FIELDS}
                FROM {self._table} t
                LEFT JOIN {self._status_table} s ON t.status_id = s.id
                WHERE t.id = %s
                """,
                (claimed["id"],)
            )
            logger.info(f"Задача {claimed['id']} выдана воркеру")
            return dict(task) if task is not None else None

    def add_agent_response(
            self,
            task_id: str,
            agent_response_id: str
    ) -> bool:
        """Добавление id ответа агента в задачу"""
        query = f"""
            UPDATE {self._table}
            SET agent_respons_ids = agent_respons_ids || %s::jsonb
            WHERE id = %s
            RETURNING id
        """
        params = (json.dumps(agent_response_id), task_id)

        with self.db.session() as db:
            result = db.fetch_one(query, params)
            return result is not None

import json
from typing import List, Dict, Any
from psycopg2.extras import Json

from .postresql import PostgresManager
from app.config import postgresql_config
from app.logs import get_logger

logger = get_logger(__name__)

class TrainingTaskManager:
    def __init__(self):
        self.db = PostgresManager(postgresql_config.URL)
        self._table = "train_models_tasks"
        self._status_tables = "task_statuses"

    def count_task(self) -> int:
        """Выводит количество имеющихся задач"""
        query = f"""
            SELECT COUNT(id)
            FROM {self._table} 
        """
        with self.db as db:
            result = db.fetch_one(query)
            return result[0] if result else 0

    def create(
            self, 
            model_id: str, 
            name: str | None = None,
            discussion_id: str | None = None
    ) -> str:
        """Создание задачи"""

        if name is None:
            pass

        query = f"""
            INSERT INTO {self._table} (model_id, name, discussion_id)
            VALUES (%s, %s, %s)
            RETURNING id
        """
        with self.db as db:
            result = db.fetch_one(query, (model_id, name, discussion_id))

            if not result:
                raise RuntimeError(f"Ошибка создания задачи для обучения модели({model_id})")
            
            task_id = result[0]
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
        params = (task_id,)
        with self.db as db:
            row = db.fetch_one(query, params)
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
        
        with self.db as db:

            # Получаем status_id
            status_result = db.fetch_one(
                f"SELECT id FROM {self._status_tables} WHERE status = %s",
                (status,)
            )

            if not status_result:
                raise ValueError(f"Неизвестный статус: {status}")

            status_id = status_result[0]

            # Задачу в финальном статусе обновлять нельзя
            # (иначе progress-апдейты воркера перетирают cancelled/completed/failed)
            current = db.fetch_one(
                f"""
                SELECT s.status FROM {self._table} t
                LEFT JOIN {self._status_tables} s ON t.status_id = s.id
                WHERE t.id = %s
                """,
                (task_id,)
            )

            if current is None:
                raise ValueError(f"Задача с ID {task_id} не найдена")

            if current[0] in ('completed', 'failed', 'cancelled'):
                raise ValueError(
                    f"Задача в финальном статусе '{current[0]}' не может быть обновлена"
                )

            # Обновляем задачу
            query = f"""
                UPDATE {self._table} 
                SET status_id = %s, 
                    status_info = %s, 
                    error_message = %s,
                    percentages = COALESCE(%s, percentages),
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                RETURNING id
            """
            
            result = db.fetch_one(query, (status_id, status_info, error, percentages, task_id))
            
            if not result:
                raise ValueError(f"Задача с ID {task_id} не найдена")
            
            logger.info(f"Статус задачи {task_id} обновлён на '{status}'")

    def add_agent_response(
        self, 
        task_id: str, 
        agent_response_id: str
    ) -> bool:
        """Добавить ID ответа агента к задаче"""
        
        with self.db as db:
            # Получаем имющиеся ID
            row = db.fetch_one(
                f"SELECT agent_respons_ids FROM {self._table} WHERE id = %s",
                (task_id,)
            )
            
            if not row:
                logger.warning(f"Задача {task_id} не найдена")
                return False
            
            # Получаем массив ID или пустой список
            response_ids = row[0] if row[0] is not None else []
            
            # Добавляем новый ID
            if agent_response_id not in response_ids:
                response_ids.append(agent_response_id)
            
            # Обновляем
            db.execute(
                f"UPDATE {self._table} SET agent_respons_ids = %s WHERE id = %s",
                (Json(response_ids), task_id)
            )
            
            logger.info(f"Добавлен ответ {agent_response_id} к задаче {task_id}")
            return True
        
    def get_task(
            self, 
            task_id: str
    ) -> Dict | None:
        """Получить информацию о задаче"""

        query = f"""
            SELECT 
                t.id,
                t.name,
                t.model_id,
                t.discussion_id,
                t.agent_respons_ids,
                t.status_id,
                s.status as status_name,
                s.description as status_description,
                t.percentages,
                t.status_info,
                t.error_message,
                t.created_at,
                t.started_at,
                t.updated_at,
                t.completed_at
            FROM {self._table} t
            LEFT JOIN {self._status_tables} s ON t.status_id = s.id
            WHERE t.id = %s
        """

        with self.db as db:
            row = db.fetch_one(query, (task_id,))
            
            if row is None:
                return None
            
            return {
                'id': row[0],
                'name': row[1],
                'model_id': row[2],
                'discussion_id': row[3],
                'agent_respons_ids': row[4],
                'status_id': row[5],
                'status': row[6],
                'status_description': row[7],
                "percentages": row[8],
                'status_info': row[9],
                'error_message': row[10],
                'created_at': row[11],
                'started_at': row[12],
                'updated_at': row[13],
                'completed_at': row[14]
            }
        
    def get_tasks(
            self
    ) -> List[Dict] | None:
        """Получить информацию о задачах"""

        query = f"""
            SELECT 
                t.id,
                t.name,
                t.model_id,
                t.discussion_id,
                t.agent_respons_ids,
                t.status_id,
                s.status as status_name,
                s.description as status_description,
                t.percentages,
                t.status_info,
                t.error_message,
                t.created_at,
                t.started_at,
                t.updated_at,
                t.completed_at
            FROM {self._table} t
            LEFT JOIN {self._status_tables} s ON t.status_id = s.id
        """

        with self.db as db:
            rows = db.fetch_all(query)
            
            if len(rows) == 0:
                return None
            
            return [
                {
                    'id': row[0],
                    'name': row[1],
                    'model_id': row[2],
                    'discussion_id': row[3],
                    'agent_respons_ids': row[4],
                    'status_id': row[5],
                    'status': row[6],
                    'status_description': row[7],
                    "percentages": row[8],
                    'status_info': row[9],
                    'error_message': row[10],
                    'created_at': row[11],
                    'started_at': row[12],
                    'updated_at': row[13],
                    'completed_at': row[14]
                }
                for row in rows
            ]

    def get_status_values(self) -> List[Dict[str, Any]]:
        """Получить все возможные значения статуса"""
        query = f"""
            SELECT id, status, description
            FROM {self._status_tables}
        """
        with self.db as db:
            rows = db.fetch_all(query)
            return [
                {
                    "id": row[0],
                    "status": row[1],
                    "description": row[2]
                }
                for row in rows
            ]

    def get_task_with_status(
        self,
        status: str
    ) -> List[Dict] | None:
        """Получить задачи c заданным статусом"""
        query = f"""
            SELECT 
                t.id, 
                t.name,
                t.model_id,
                s.status,
                t.status_id,
                s.description as status_description,
                t.discussion_id,
                t.agent_respons_ids,
                t.percentages,
                t.status_info,
                t.error_message,
                t.created_at,
                t.started_at,
                t.updated_at,
                t.completed_at
            FROM {self._table} t 
            LEFT JOIN {self._status_tables} s ON t.status_id = s.id
            WHERE s.status = %s
            ORDER BY created_at ASC
        """

        with self.db as db:
            rows = db.fetch_all(query, (status,))

            if len(rows) == 0:
                return None

            columns = [
                'id', 'name', 'model_id', 'status', 'status_id', 'status_description',
                'discussion_id', 'agent_respons_ids', 'percentages',
                'status_info', 'error_message',
                'created_at', 'started_at', 'updated_at', 
                'completed_at'
            ]

            return [dict(zip(columns, row)) for row in rows]

    def get_next_task(self) -> Dict[str, Any] | None:
        "Получить первую задачу в очереди на выполнение"
        tasks = self.get_task_with_status('waiting')

        if tasks is None:
            return None
        return tasks[0]

    def add_agent_respons(
            self, 
            task_id: str, 
            agent_respons_id: str
        ):
        """Добавление id ответа агента в задачу"""
        query = f"""
            UPDATE {self._table}
            SET agent_respons_ids = agent_respons_ids || %s::jsonb
            WHERE id = %s
            RETURNING id
        """
        params = (json.dumps(agent_respons_id), task_id)
        
        with self.db as db:
            result = db.fetch_one(query, params)
            return result is not None
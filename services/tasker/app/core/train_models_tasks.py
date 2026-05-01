from typing import List, Dict
from psycopg2.extras import Json

from .postresql import PostgresManager
from app.config import postgresql_config
from app.logs import get_logger

logger = get_logger(__name__)

class TrainingTaskManager:
    def __init__(self):
        self.db = PostgresManager(postgresql_config.URL)
        self._table = "train_models_tasks"

    def create(
            self, 
            model_id: str, 
            discussion_id: str | None = None
    ) -> str:
        """Создание задачи"""
        query = f"""
            INSERT INTO {self._table} (model_id, discussion_id)
            VALUES (%s, %s)
            RETURNING id
        """
        with self.db as db:
            result = db.fetch_one(query, (model_id, discussion_id))

            if not result:
                raise RuntimeError(f"Ошибка создания задачи для обучения модели({model_id})")
            
            task_id = result[0]
            logger.info(f"✅ Создана задача {task_id} для модели {model_id}")
            return task_id

    def update_status(
            self, 
            task_id: str, 
            status: str, 
            error: str | None = None
    ):
        """Обновить статус"""
        query = f"""
            UPDATE {self._table} 
            SET status = %s, 
                error_message = %s,
                completed_at = CASE WHEN %s = 'completed' THEN CURRENT_TIMESTAMP END,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
        """
        with self.db as db:
            db.execute(query, (status, error, status, task_id))

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
    
    def get_pending(self) -> List[Dict]:
        """Получить все pending задачи"""
        query = f"""
            SELECT id, model_id, discussion_id, agent_respons_ids
            FROM {self._table} 
            WHERE status = 'pending'
            ORDER BY created_at ASC
        """
        with self.db as db:
            rows = db.fetch_all(query)
            return [dict(zip(['id', 'model_id', 'discussion_id', 'agent_respons_ids'], row)) for row in rows]

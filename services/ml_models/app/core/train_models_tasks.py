from typing import List, Dict, Any
from psycopg2.extras import Json

from .postresql import PostgresManager
from app.config import postgresql_config
from app.logs import get_logger

logger = get_logger(__name__)

class MlModelsManager:
    def __init__(self):
        self.db = PostgresManager(postgresql_config.URL)
        self._models_table = "ml_models"
        self._statuses_models_table = "ml_model_statuses"

    def get_statuses_info(self) -> List[Dict[str, Any]]:
        """Получить список возможных статусов"""
        query = f"""
            SELECT 
                id,
                status,
                description
            FROM {self._statuses_models_table} 
        """
        with self.db as db:
            
            db_response = db.fetch_all(query)
            if len(db_response) == 0:
                logger.warning("Не найдены статусы")
                return []
            
            results = [
                {
                    "id": result[0],
                    "status": result[1],
                    "description": result[2]
                } 
                for result in db_response
            ]

            return results

    def count_models(self) -> int:
        """Выводит количество имеющихся моделей"""
        query = f"""
            SELECT COUNT(id)
            FROM {self._models_table} 
        """
        with self.db as db:
            result = db.fetch_one(query)
            return result[0] if result else 0

    def create(
            self, 
            name: str,
            model_type: str,
            description: str | None,
            classes: List[str],
            train_params: dict,
    ) -> str:
        """Создание модели"""

        query = f"""
            INSERT INTO {self._models_table} (name, model_type, description, classes, train_params)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """
        with self.db as db:
            result = db.fetch_one(query, (name, model_type, description, Json(classes), Json(train_params)))

            if not result:
                raise RuntimeError(f"Ошибка создания модели (name={name}, model_type={model_type})")
            
            model_id = result[0]
            logger.info(f"✅ Создана модель {model_id}")
            return model_id
        
    def delete(
        self,
        model_id: str,
    ) -> bool:
        """Удаление модели"""
        query = f"""
            DELETE FROM {self._models_table}
            WHERE id = %s
            RETURNING id
        """
        params = (model_id,)
        with self.db as db:
            row = db.fetch_one(query, params)
            return row is not None

    def update_model(
            self, 
            model_id: str, 
            model_type: str | None = None, 
            description: str | None = None
    ):
        """Обновить статус"""
        query = f"""
            UPDATE {self._models_table} 
            SET model_type = %s, 
                description = %s, 
            WHERE id = %s
        """
        with self.db as db:
            db.execute(query, (model_type, description, model_id))
        
    def get_model(
            self, 
            model_id: str
    ) -> Dict | None:
        """Получить информацию о модели"""

        query = f"""
            SELECT 
                id,
                name,
                model_type,
                description,
                classes,
                train_params,
                created_at
            FROM {self._models_table}
            WHERE id = %s
        """
        with self.db as db:
            row = db.fetch_one(query, (model_id,))
            
            if row is None:
                return None
            
            return {
                'id': row[0],
                'name': row[1],
                'model_type': row[2],
                'description': row[3],
                'classes': row[4],
                'train_params': row[5],
                'creatred_at': row[6]
            }

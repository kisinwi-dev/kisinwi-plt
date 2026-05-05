from typing import List, Dict, Any
from psycopg2.extras import Json
from psycopg2 import DatabaseError, IntegrityError

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
            version: str,
            model_type: str,
            classes: list,
            dataset_id: str ,
            dataset_version_id: str,
            train_params: dict,
            description: str | None = None,
            framework: str | None = None,
            framework_version: str | None = None,
    ) -> str:
        """Создание модели"""

        fields = [
            "name", "version", "model_type", 
            "classes", "dataset_id", "dataset_version_id", "train_params"
        ]

        values = [
            name, version, model_type,
            Json(classes), dataset_id, dataset_version_id, Json(train_params)
        ]

        if description is not None:
            fields.append("description")
            values.append(description)
        
        if framework is not None:
            fields.append("framework")
            values.append(framework)
        
        if framework_version is not None:
            fields.append("framework_version")
            values.append(framework_version)
        
        fields_str = ", ".join(fields)
        placeholders = ", ".join(["%s"] * len(values))

        query = f"""
            INSERT INTO {self._models_table} ({fields_str})
            VALUES ({placeholders})
            RETURNING id
        """
        with self.db as db:
            try:
                result = db.fetch_one(query, tuple(values))
                
                if not result:
                    raise RuntimeError(f"Не удалось создать модель (name={name}, version={version})")
                
                model_id = result[0]
                logger.info(f"✅ Создана модель {model_id} (name={name}, version={version})")
                return model_id
            except IntegrityError as e:
                if "unique_model_name_version" in str(e):
                    logger.error(f"Модель с именем '{name}' и версией '{version}' уже существует")
                    raise ValueError(f"Модель с именем '{name}' и версией '{version}' уже существует")
                
                logger.error(f"Ошибка целостности данных: {e}")
                raise RuntimeError(f"Ошибка целостности данных: {e}")

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

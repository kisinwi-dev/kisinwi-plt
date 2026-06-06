import uuid
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
            result = db.fetch_one(query, params)

            if result is None:
                logger.warning(f"Не удалось найти модель '{model_id}' для удаления")
                return False

            logger.info(f"✅ Удалена модель '{model_id}'")
            return True

    def update_model(
            self, 
            model_id: str, 
            update_data: Dict[str, Any]
    ):
        """Обновить информацию о модели"""

        data = {
            k: v for k, v in update_data.items()
            if v is not None
        }

        # Проверка существования модели
        if self.get_model(str(model_id)) is None:
            raise ValueError("")

        # Обработка статуса
        if data.get("status"):
            statuses = self.get_statuses_info()
            status_found = False

            for status in statuses:
                if status["status"] == data["status"]:
                    data["status_id"] = status["id"]
                    status_found = True
                    break

            if status_found:
                logger.debug("Значение статуса переведено в id статуса")
            else:
                logger.error(f"Неизвестный статус '{data["status"]}'")
                raise ValueError(f"Неизвестный статус '{data["status"]}'")

            del data["status"]

        # Формирование запрос
        set_clauses = []
        values = []
        for key, val in data.items():
            set_clauses.append(f"{key} = %s")

            # JSON поле
            if key in ("classes", "train_params"):
                val = Json(val)

            values.append(val)

        values.append(str(model_id)) # для where

        query = f"""
            UPDATE {self._models_table}
            SET {", ".join(set_clauses)}
            WHERE id = %s
            RETURNING id
        """

        with self.db as db:
            try:
                result = db.fetch_one(query, tuple(values))
                
                if result is None:
                    logger.error(f"Модель '{model_id}' не обновлена")
                    return False

                logger.info(f"✅ Обновлена модель '{model_id}'")
                return True

            except Exception as e:
                logger.error(f"Ошибка при обновлении модели {model_id}: {e}", exc_info=True)
                raise RuntimeError(f"Не удалось обновить модель: {e}")

    def get_model(
        self,
        model_id: str | None = None,
        dataset_id: str | None = None,
        status: str | None = None
    ) -> List[Dict[str, Any]] | None:
        """
        Получить полную информацию о моделях.

        Без фильтров возвращает все модели (свежие сверху). Опционально
        фильтрует по model_id, dataset_id и/или статусу модели.
        """

        query = f"""
            SELECT
                m.id,
                m.name,
                m.version,
                m.model_type,
                s.status as status,
                m.description,
                m.metrics_result,
                m.classes,
                m.train_params,
                m.created_at,
                m.dataset_id,
                m.dataset_version_id,
                m.framework,
                m.framework_version
            FROM {self._models_table} m
            LEFT JOIN {self._statuses_models_table} s ON m.status_id = s.id
        """

        conditions = []
        params: list = []
        if model_id:
            conditions.append("m.id = %s")
            params.append(model_id)
        if dataset_id:
            conditions.append("m.dataset_id = %s")
            params.append(dataset_id)
        if status:
            conditions.append("s.status = %s")
            params.append(status)

        if conditions:
            query += "WHERE " + " AND ".join(conditions) + "\n"

        query += "ORDER BY m.created_at DESC\n"

        with self.db as db:
            rows = db.fetch_all(query, tuple(params) if params else None)

            if len(rows) == 0:
                return None

            models = [
                self._row_full_info_model_in_dict(row)
                for row in rows
            ]

            return models
    
    def _row_full_info_model_in_dict(self, row: tuple):
        """Преобразовать всю строчку из соединённой таблицы  """
        return {
            'id': str(row[0]),
            'name': row[1],
            'version': row[2],  
            'model_type': row[3],
            'status': row[4],
            'description': row[5],
            'metrics_result': row[6],
            'classes': row[7],
            'train_params': row[8],
            'created_at': row[9],
            'dataset_id': str(row[10]),
            'dataset_version_id': row[11],
            'framework': row[12],
            'framework_version': row[13]
        }
    
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

    def count_model(
        self,
        dataset_id: str | None = None,
        status: str | None = None,
        name: str | None = None
    ) -> int:
        """Количество моделей с учётом тех же фильтров, что и get_model (для пагинации)"""
        query = f"""
            SELECT COUNT(m.id)
            FROM {self._models_table} m
            LEFT JOIN {self._statuses_models_table} s ON m.status_id = s.id
        """

        conditions = []
        params: list = []
        if dataset_id:
            conditions.append("m.dataset_id = %s")
            params.append(dataset_id)
        if status:
            conditions.append("s.status = %s")
            params.append(status)
        if name:
            conditions.append("m.name ILIKE %s")
            params.append(f"%{name}%")

        if conditions:
            query += "WHERE " + " AND ".join(conditions) + "\n"

        with self.db as db:
            result = db.fetch_one(query, tuple(params) if params else None)
            return result[0] if result else 0

    def get_statistics(self) -> Dict[str, Any]:
        """
        Статистика по моделям: общее количество и распределение по статусам.

        Один запрос с группировкой по статусам (LEFT JOIN, чтобы статусы без
        моделей тоже попадали с count=0).
        """
        query = f"""
            SELECT s.id, s.status, s.description, COUNT(m.id)
            FROM {self._statuses_models_table} s
            LEFT JOIN {self._models_table} m ON m.status_id = s.id
            GROUP BY s.id, s.status, s.description
            ORDER BY s.id
        """
        with self.db as db:
            rows = db.fetch_all(query)

        by_status = [
            {
                "status": {"id": row[0], "status": row[1], "description": row[2]},
                "count": row[3],
            }
            for row in rows
        ]
        total = sum(item["count"] for item in by_status)

        return {"total": total, "by_status": by_status}

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

    def delete_by_name(self, name: str) -> list[str]:
        """Удаление всех версий модели по имени. Возвращает ID удалённых записей."""
        query = f"""
            DELETE FROM {self._models_table}
            WHERE name = %s
            RETURNING id
        """
        with self.db as db:
            rows = db.fetch_all(query, (name,))
            ids = [str(row[0]) for row in rows] if rows else []
            if not ids:
                logger.warning(f"Не найдено моделей с именем '{name}' для удаления")
            else:
                logger.info(f"✅ Удалено {len(ids)} версий модели '{name}'")
            return ids

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
        status: str | None = None,
        name: str | None = None,
        limit: int | None = None,
        offset: int = 0
    ) -> List[Dict[str, Any]] | None:
        """
        Получить полную информацию о моделях.

        Без фильтров возвращает все модели (свежие сверху). Опционально
        фильтрует по model_id, dataset_id, статусу и/или имени модели (name —
        для получения всех версий модели). Если задан limit — применяется
        пагинация (LIMIT/OFFSET).
        """

        query = f"""
            SELECT
                m.id,
                m.name,
                m.version,
                m.model_type,
                s.status as status,
                m.description,
                m.metrics_report,
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
        if name:
            conditions.append("m.name ILIKE %s")
            params.append(f"%{name}%")

        if conditions:
            query += "WHERE " + " AND ".join(conditions) + "\n"

        query += "ORDER BY m.created_at DESC\n"

        if limit is not None:
            query += "LIMIT %s OFFSET %s\n"
            params.append(limit)
            params.append(offset)

        with self.db as db:
            rows = db.fetch_all(query, tuple(params) if params else None)

            if len(rows) == 0:
                return None

            models = [
                self._row_full_info_model_in_dict(row)
                for row in rows
            ]

            return models
    
    def get_grouped_models(
        self,
        dataset_id: str | None = None,
        status: str | None = None,
        name: str | None = None,
        limit: int | None = None,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Модели сгруппированные по имени, версии отсортированы по убыванию.
        Пагинация применяется к уникальным именам, а не к отдельным моделям.
        """
        base_join = f"""
            FROM {self._models_table} m
            LEFT JOIN {self._statuses_models_table} s ON m.status_id = s.id
        """

        conditions: list = []
        filter_params: list = []
        if dataset_id:
            conditions.append("m.dataset_id = %s")
            filter_params.append(dataset_id)
        if status:
            conditions.append("s.status = %s")
            filter_params.append(status)
        if name:
            conditions.append("m.name ILIKE %s")
            filter_params.append(f"%{name}%")

        filter_where = ("WHERE " + " AND ".join(conditions) + "\n") if conditions else ""

        with self.db as db:
            # 1. Количество уникальных имён
            count_query = f"SELECT COUNT(DISTINCT m.name) {base_join} {filter_where}"
            total_result = db.fetch_one(count_query, tuple(filter_params) if filter_params else None)
            total = total_result[0] if total_result else 0

            if total == 0:
                return {"groups": [], "total": 0, "limit": limit, "offset": offset}

            # 2. Имена для текущей страницы
            names_query = f"SELECT DISTINCT m.name {base_join} {filter_where} ORDER BY m.name ASC"
            names_params = list(filter_params)
            if limit is not None:
                names_query += " LIMIT %s OFFSET %s"
                names_params.extend([limit, offset])

            name_rows = db.fetch_all(names_query, tuple(names_params) if names_params else None)
            names = [row[0] for row in name_rows]

            if not names:
                return {"groups": [], "total": total, "limit": limit, "offset": offset}

            # 3. Все версии для отобранных имён
            version_conditions = conditions + ["m.name = ANY(%s)"]
            versions_where = "WHERE " + " AND ".join(version_conditions) + "\n"

            versions_query = f"""
                SELECT
                    m.id, m.name, m.version, m.model_type, s.status,
                    m.description, m.metrics_report, m.classes, m.train_params,
                    m.created_at, m.dataset_id, m.dataset_version_id, m.framework, m.framework_version
                {base_join}
                {versions_where}
                ORDER BY m.name ASC, m.version DESC
            """
            versions_params = filter_params + [names]
            version_rows = db.fetch_all(versions_query, tuple(versions_params))

        groups_map: Dict[str, list] = {name: [] for name in names}
        for row in version_rows:
            row_name = row[1]
            if row_name in groups_map:
                groups_map[row_name].append(self._row_full_info_model_in_dict(row))

        groups = [
            {"name": name, "versions": groups_map[name]}
            for name in names
            if groups_map[name]
        ]

        return {"groups": groups, "total": total, "limit": limit, "offset": offset}

    def _row_full_info_model_in_dict(self, row: tuple):
        """Преобразовать всю строчку из соединённой таблицы  """
        return {
            'id': str(row[0]),
            'name': row[1],
            'version': row[2],  
            'model_type': row[3],
            'status': row[4],
            'description': row[5],
            'metrics_report': row[6],
            'classes': row[7],
            'train_params': row[8],
            'created_at': row[9],
            'dataset_id': str(row[10]),
            'dataset_version_id': row[11],
            'framework': row[12],
            'framework_version': row[13]
        }
    
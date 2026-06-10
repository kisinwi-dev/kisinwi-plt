from typing import List, Dict, Any
from psycopg2 import IntegrityError

from .postresql import PostgresManager
from .versions import row_version_in_dict, VERSION_SELECT_FIELDS
from app.config import postgresql_config
from app.logs import get_logger

logger = get_logger(__name__)

class ModelsManager:
    """Работа с родительской сущностью модели (таблица models)"""

    def __init__(self):
        self.db = PostgresManager(postgresql_config.URL)
        self._models_table = "models"
        self._versions_table = "model_versions"
        self._statuses_table = "ml_model_statuses"

    def create(
        self,
        name: str,
        description: str | None = None,
    ) -> str:
        """Создание модели. ValueError при дубле имени."""
        query = f"""
            INSERT INTO {self._models_table} (name, description)
            VALUES (%s, %s)
            RETURNING id
        """
        with self.db as db:
            try:
                result = db.fetch_one(query, (name, description))

                if not result:
                    raise RuntimeError(f"Не удалось создать модель (name={name})")

                model_id = str(result[0])
                logger.info(f"✅ Создана модель {model_id} (name={name})")
                return model_id
            except IntegrityError as e:
                if "unique_model_name" in str(e):
                    logger.error(f"Модель с именем '{name}' уже существует")
                    raise ValueError(f"Модель с именем '{name}' уже существует")

                logger.error(f"Ошибка целостности данных: {e}")
                raise RuntimeError(f"Ошибка целостности данных: {e}")

    def update(self, model_id: str, update_data: Dict[str, Any]) -> bool:
        """Обновить имя/описание модели. ValueError при дубле имени."""
        data = {
            k: v for k, v in update_data.items()
            if v is not None and k in ("name", "description")
        }

        set_clauses = [f"{key} = %s" for key in data]
        values = list(data.values())
        values.append(str(model_id))

        query = f"""
            UPDATE {self._models_table}
            SET {", ".join(set_clauses)}
            WHERE id = %s
            RETURNING id
        """

        with self.db as db:
            try:
                result = db.fetch_one(query, tuple(values))
            except IntegrityError as e:
                if "unique_model_name" in str(e):
                    raise ValueError(f"Модель с именем '{data.get('name')}' уже существует")
                raise RuntimeError(f"Ошибка целостности данных: {e}")

            if result is None:
                logger.error(f"Модель '{model_id}' не обновлена")
                return False

            logger.info(f"✅ Обновлена модель '{model_id}'")
            return True

    def delete(self, model_id: str) -> List[str] | None:
        """
        Удаление модели со всеми версиями.

        Возвращает id удалённых версий (для очистки файлов на диске)
        или None, если модель не найдена.
        """
        versions_query = f"""
            SELECT id FROM {self._versions_table}
            WHERE model_id = %s
        """
        delete_query = f"""
            DELETE FROM {self._models_table}
            WHERE id = %s
            RETURNING id
        """
        with self.db as db:
            # id версий собираем до удаления — CASCADE снесёт строки
            version_rows = db.fetch_all(versions_query, (model_id,))
            version_ids = [str(row[0]) for row in version_rows]

            result = db.fetch_one(delete_query, (model_id,))

            if result is None:
                logger.warning(f"Не удалось найти модель '{model_id}' для удаления")
                return None

            logger.info(f"✅ Удалена модель '{model_id}' ({len(version_ids)} версий)")
            return version_ids

    def get_models(
        self,
        name: str | None = None,
        status: str | None = None,
        dataset_id: str | None = None,
        limit: int | None = None,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Модели с вложенными версиями (версии по убыванию).

        Пагинация применяется к моделям. Фильтры status/dataset_id отбирают
        модели, у которых есть хотя бы одна подходящая версия; во вложенном
        списке остаются только подходящие версии.
        """
        version_conditions: list = []
        filter_params: list = []
        if dataset_id:
            version_conditions.append("v.dataset_id = %s")
            filter_params.append(dataset_id)
        if status:
            version_conditions.append("s.status = %s")
            filter_params.append(status)

        model_conditions = list(version_conditions)
        model_params = list(filter_params)
        if name:
            model_conditions.append("m.name ILIKE %s")
            model_params.append(f"%{name}%")

        # Если есть фильтры по версиям — модель должна иметь хотя бы одну подходящую
        version_filtering = bool(version_conditions)
        join = f"""
            FROM {self._models_table} m
            {"JOIN" if version_filtering else "LEFT JOIN"} {self._versions_table} v ON v.model_id = m.id
            LEFT JOIN {self._statuses_table} s ON v.status_id = s.id
        """
        model_where = ("WHERE " + " AND ".join(model_conditions) + "\n") if model_conditions else ""

        with self.db as db:
            # 1. Количество подходящих моделей
            count_query = f"SELECT COUNT(DISTINCT m.id) {join} {model_where}"
            total_result = db.fetch_one(count_query, tuple(model_params) if model_params else None)
            total = total_result[0] if total_result else 0

            if total == 0:
                return {"models": [], "total": 0, "limit": limit, "offset": offset}

            # 2. Модели для текущей страницы
            models_query = f"""
                SELECT DISTINCT m.id, m.name, m.description, m.created_at
                {join} {model_where}
                ORDER BY m.name ASC
            """
            models_params = list(model_params)
            if limit is not None:
                models_query += " LIMIT %s OFFSET %s"
                models_params.extend([limit, offset])

            model_rows = db.fetch_all(models_query, tuple(models_params) if models_params else None)

            if not model_rows:
                return {"models": [], "total": total, "limit": limit, "offset": offset}

            model_ids = [str(row[0]) for row in model_rows]

            # 3. Версии для отобранных моделей (с учётом фильтров по версиям)
            versions_where_parts = version_conditions + ["v.model_id = ANY(%s::uuid[])"]
            versions_query = f"""
                SELECT {VERSION_SELECT_FIELDS}
                FROM {self._versions_table} v
                JOIN {self._models_table} m ON v.model_id = m.id
                LEFT JOIN {self._statuses_table} s ON v.status_id = s.id
                WHERE {" AND ".join(versions_where_parts)}
                ORDER BY v.version DESC
            """
            version_rows = db.fetch_all(versions_query, tuple(filter_params + [model_ids]))

        versions_map: Dict[str, list] = {model_id: [] for model_id in model_ids}
        for row in version_rows:
            version = row_version_in_dict(row)
            versions_map[version["model_id"]].append(version)

        models = [
            {
                "id": str(row[0]),
                "name": row[1],
                "description": row[2],
                "created_at": row[3],
                "versions": versions_map[str(row[0])],
            }
            for row in model_rows
        ]

        return {"models": models, "total": total, "limit": limit, "offset": offset}

    def get_by_id(self, model_id: str) -> Dict[str, Any] | None:
        """Модель с версиями по id"""
        return self._get_one("m.id = %s::uuid", model_id)

    def get_by_name(self, name: str) -> Dict[str, Any] | None:
        """Модель с версиями по точному имени"""
        return self._get_one("m.name = %s", name)

    def _get_one(self, condition: str, param: str) -> Dict[str, Any] | None:
        model_query = f"""
            SELECT id, name, description, created_at
            FROM {self._models_table} m
            WHERE {condition}
        """
        with self.db as db:
            row = db.fetch_one(model_query, (param,))
            if row is None:
                return None

            versions_query = f"""
                SELECT {VERSION_SELECT_FIELDS}
                FROM {self._versions_table} v
                JOIN {self._models_table} m ON v.model_id = m.id
                LEFT JOIN {self._statuses_table} s ON v.status_id = s.id
                WHERE v.model_id = %s
                ORDER BY v.version DESC
            """
            version_rows = db.fetch_all(versions_query, (str(row[0]),))

        return {
            "id": str(row[0]),
            "name": row[1],
            "description": row[2],
            "created_at": row[3],
            "versions": [row_version_in_dict(v) for v in version_rows],
        }

    def get_statistics(self) -> Dict[str, Any]:
        """
        Статистика: количество моделей, версий и распределение версий по статусам.

        LEFT JOIN — чтобы статусы без версий тоже попадали с count=0.
        """
        statuses_query = f"""
            SELECT s.id, s.status, s.description, COUNT(v.id)
            FROM {self._statuses_table} s
            LEFT JOIN {self._versions_table} v ON v.status_id = s.id
            GROUP BY s.id, s.status, s.description
            ORDER BY s.id
        """
        models_query = f"SELECT COUNT(id) FROM {self._models_table}"
        with self.db as db:
            rows = db.fetch_all(statuses_query)
            models_count = db.fetch_one(models_query)

        by_status = [
            {
                "status": {"id": row[0], "status": row[1], "description": row[2]},
                "count": row[3],
            }
            for row in rows
        ]
        total_versions = sum(item["count"] for item in by_status)

        return {
            "total_models": models_count[0] if models_count else 0,
            "total_versions": total_versions,
            "by_status": by_status,
        }

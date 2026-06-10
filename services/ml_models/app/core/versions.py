from typing import List, Dict, Any, Tuple
from psycopg2.extras import Json
from psycopg2 import IntegrityError
from psycopg2 import errors as pg_errors

from .postresql import PostgresManager
from app.config import postgresql_config
from app.logs import get_logger

logger = get_logger(__name__)

# Поля версии в порядке row_version_in_dict (v — model_versions, m — models, s — statuses)
VERSION_SELECT_FIELDS = """
    v.id, v.model_id, m.name, m.description, v.version, v.model_type,
    s.status, v.metrics_report, v.classes, v.train_params, v.created_at,
    v.dataset_id, v.dataset_version_id, v.framework, v.framework_version
"""

def row_version_in_dict(row: tuple) -> Dict[str, Any]:
    """Преобразовать строку version-запроса (VERSION_SELECT_FIELDS) в словарь"""
    return {
        "id": str(row[0]),
        "model_id": str(row[1]),
        "name": row[2],
        "description": row[3],
        "version": row[4],
        "model_type": row[5],
        "status": row[6],
        "metrics_report": row[7],
        "classes": row[8],
        "train_params": row[9],
        "created_at": row[10],
        "dataset_id": str(row[11]),
        "dataset_version_id": row[12],
        "framework": row[13],
        "framework_version": row[14],
    }

class VersionsManager:
    """Работа с версиями модели (таблица model_versions)"""

    def __init__(self):
        self.db = PostgresManager(postgresql_config.URL)
        self._versions_table = "model_versions"
        self._models_table = "models"
        self._statuses_table = "ml_model_statuses"

    def get_statuses_info(self) -> List[Dict[str, Any]]:
        """Получить список возможных статусов"""
        query = f"""
            SELECT
                id,
                status,
                description
            FROM {self._statuses_table}
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

    def create(
        self,
        model_id: str,
        model_type: str,
        classes: list,
        dataset_id: str,
        dataset_version_id: str,
        train_params: dict,
        framework: str | None = None,
        framework_version: str | None = None,
    ) -> Tuple[str, int]:
        """
        Создание версии: номер назначается сервером (MAX+1 по модели).

        Возвращает (version_id, version). ValueError, если родителя нет.
        """
        query = f"""
            INSERT INTO {self._versions_table} (
                model_id, version, model_type, classes,
                dataset_id, dataset_version_id, train_params,
                framework, framework_version
            )
            VALUES (
                %s,
                (SELECT COALESCE(MAX(version), 0) + 1 FROM {self._versions_table} WHERE model_id = %s),
                %s, %s, %s, %s, %s, %s, %s
            )
            RETURNING id, version
        """
        params = (
            model_id, model_id, model_type, Json(classes),
            dataset_id, dataset_version_id, Json(train_params),
            framework, framework_version,
        )

        # Один retry: параллельная вставка может занять тот же номер версии
        for attempt in (1, 2):
            with self.db as db:
                try:
                    result = db.fetch_one(query, params)

                    if not result:
                        raise RuntimeError(f"Не удалось создать версию модели '{model_id}'")

                    version_id, version = str(result[0]), result[1]
                    logger.info(f"✅ Создана версия {version} ({version_id}) модели '{model_id}'")
                    return version_id, version
                except IntegrityError as e:
                    if isinstance(e, pg_errors.ForeignKeyViolation):
                        logger.error(f"Модель '{model_id}' не найдена")
                        raise ValueError(f"Модель с ID {model_id} не найдена")

                    if "unique_model_version" in str(e) and attempt == 1:
                        logger.warning(f"Конфликт номера версии для модели '{model_id}', повтор")
                        continue

                    logger.error(f"Ошибка целостности данных: {e}")
                    raise RuntimeError(f"Ошибка целостности данных: {e}")

        raise RuntimeError(f"Не удалось создать версию модели '{model_id}'")

    def delete_version(self, version_id: str) -> bool:
        """Удаление версии"""
        query = f"""
            DELETE FROM {self._versions_table}
            WHERE id = %s
            RETURNING id
        """
        with self.db as db:
            result = db.fetch_one(query, (version_id,))

            if result is None:
                logger.warning(f"Не удалось найти версию '{version_id}' для удаления")
                return False

            logger.info(f"✅ Удалена версия '{version_id}'")
            return True

    def update_version(
        self,
        version_id: str,
        update_data: Dict[str, Any]
    ) -> bool:
        """Обновить информацию о версии"""

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

        # Формирование запроса
        set_clauses = []
        values = []
        for key, val in data.items():
            set_clauses.append(f"{key} = %s")

            # JSON поле
            if key in ("classes", "train_params"):
                val = Json(val)

            values.append(val)

        values.append(str(version_id)) # для where

        query = f"""
            UPDATE {self._versions_table}
            SET {", ".join(set_clauses)}
            WHERE id = %s
            RETURNING id
        """

        with self.db as db:
            try:
                result = db.fetch_one(query, tuple(values))

                if result is None:
                    logger.error(f"Версия '{version_id}' не обновлена")
                    return False

                logger.info(f"✅ Обновлена версия '{version_id}'")
                return True

            except Exception as e:
                logger.error(f"Ошибка при обновлении версии {version_id}: {e}", exc_info=True)
                raise RuntimeError(f"Не удалось обновить версию: {e}")

    def get_version(self, version_id: str) -> Dict[str, Any] | None:
        """Получить одну версию с полями родителя"""
        query = f"""
            SELECT {VERSION_SELECT_FIELDS}
            FROM {self._versions_table} v
            JOIN {self._models_table} m ON v.model_id = m.id
            LEFT JOIN {self._statuses_table} s ON v.status_id = s.id
            WHERE v.id = %s::uuid
        """
        with self.db as db:
            row = db.fetch_one(query, (version_id,))

        if row is None:
            return None

        return row_version_in_dict(row)

    def get_versions(
        self,
        name: str | None = None,
        status: str | None = None,
        dataset_id: str | None = None,
        model_id: str | None = None,
        limit: int | None = None,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Плоский список версий (свежие сверху) с опциональными фильтрами
        и пагинацией (LIMIT/OFFSET при заданном limit).
        """
        query = f"""
            SELECT {VERSION_SELECT_FIELDS}
            FROM {self._versions_table} v
            JOIN {self._models_table} m ON v.model_id = m.id
            LEFT JOIN {self._statuses_table} s ON v.status_id = s.id
        """

        conditions, params = self._filters(name, status, dataset_id, model_id)
        if conditions:
            query += "WHERE " + " AND ".join(conditions) + "\n"

        query += "ORDER BY v.created_at DESC\n"

        if limit is not None:
            query += "LIMIT %s OFFSET %s\n"
            params.append(limit)
            params.append(offset)

        with self.db as db:
            rows = db.fetch_all(query, tuple(params) if params else None)

        return [row_version_in_dict(row) for row in rows]

    def count_versions(
        self,
        name: str | None = None,
        status: str | None = None,
        dataset_id: str | None = None,
        model_id: str | None = None,
    ) -> int:
        """Количество версий с теми же фильтрами, что и get_versions (для пагинации)"""
        query = f"""
            SELECT COUNT(v.id)
            FROM {self._versions_table} v
            JOIN {self._models_table} m ON v.model_id = m.id
            LEFT JOIN {self._statuses_table} s ON v.status_id = s.id
        """

        conditions, params = self._filters(name, status, dataset_id, model_id)
        if conditions:
            query += "WHERE " + " AND ".join(conditions) + "\n"

        with self.db as db:
            result = db.fetch_one(query, tuple(params) if params else None)
            return result[0] if result else 0

    def _filters(
        self,
        name: str | None,
        status: str | None,
        dataset_id: str | None,
        model_id: str | None,
    ) -> Tuple[list, list]:
        conditions: list = []
        params: list = []
        if name:
            conditions.append("m.name ILIKE %s")
            params.append(f"%{name}%")
        if status:
            conditions.append("s.status = %s")
            params.append(status)
        if dataset_id:
            conditions.append("v.dataset_id = %s")
            params.append(dataset_id)
        if model_id:
            conditions.append("v.model_id = %s::uuid")
            params.append(model_id)
        return conditions, params

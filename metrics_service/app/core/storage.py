from typing import Optional
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

from app.api.schemes import *
from app.config import mongodb_config
from app.logs import get_logger

logger = get_logger(__name__)

class CVMetricManager:
    def __init__(
            self
    ) -> None:
        self.url = mongodb_config.URL_METRIC
        self.database_name = mongodb_config.DATABASE_METRIC
        self.collection_cv = mongodb_config.COLLECTION_CV
        self.client: MongoClient
        self.collection: Collection
        self.create_index()

    def create_index(self):
        """Создание индексов"""

        # __WARNING__ постоянно создаёт индексы после перезагрузки контейнера
        # ТРЕБУЕТСЯ ИСПРАВИТЬ ЭТО В БУДУЩЕМ

        try:
            # Коллекция метрик CV
            self.client = MongoClient(self.url)
            self.db = self.client[self.database_name]
            self.collection = self.db[self.collection_cv]

            self.collection.create_index('task_id', unique=True)
            self.collection.create_index([('task_id', 1), ('metrics.metric', 1)])
            logger.debug("✅ Индексы созданы")
        except PyMongoError as e:
            logger.error(f"Ошибка создания индексов: {e}")

    def connect(self):
        """Подключение к MongoDB"""
        try:
            # Заходим в коллекцию для работы с CV метриками
            self.client = MongoClient(self.url)
            self.db = self.client[self.database_name]
            self.collection = self.db[self.collection_cv]

            logger.debug(f"🟩 {self.database_name} подключена")
        except PyMongoError as e:
            logger.error(f"😡 Не удалось подключиться к {self.database_name} : {e}")
            raise
    
    def disconnect(self):
        """Отключение от MongoDB"""
        if self.client:
            self.client.close()
            logger.debug(f"⚠️ Соединение с {self.database_name} закрыто")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False

    def add_metric(
            self, 
            metric: MetricAdd
    ) -> bool:
        """Добавление новой метрики для задачи"""
        try:
            # Поиск задачи
            task_doc = self.task_exists(metric.task_id)

            if task_doc:
                # Задача существует
                result = self.collection.update_one(
                    {
                        'task_id': metric.task_id,
                        'metrics.metric': metric.name
                    },
                    {
                        '$push': {
                            f'metrics.$.values': metric.value,
                        }
                    }
                )

                # Если метрика не найдена, добавляем новую
                if result.matched_count == 0:
                    new_metric = {
                        'metric': metric.name,
                        'values': [metric.value],
                    }
                    self.collection.update_one(
                        {'task_id': metric.task_id},
                        {
                            '$push': {'metrics': new_metric}
                        }
                    )
            else:

                # Новая задача - создаем документ
                new_task = {
                    'task_id': metric.task_id,
                    'metrics': [{
                        'metric': metric.name,
                        'values': [metric.value],
                    }]
                }
                self.collection.insert_one(new_task)

            logger.debug(f"➕ Добавлена метрика {metric.name}={metric.value} для задачи {metric.task_id}")
            return True

        except PyMongoError as e:
            logger.error(f"😡 Ошибка добавления метрики: {e}")
            return False

    def get_task_metrics(
        self, 
        task_id: str
    ) -> Optional[TaskMetrics]:
        """Получение всех метрик для задачи"""
        try:
            task_doc = self.collection.find_one({'task_id': task_id})
            
            if task_doc:
                return TaskMetrics(
                    task_id=task_doc['task_id'],
                    metrics=[
                        MetricData(
                            name=m['metric'],
                            values=m['values'],
                        )
                        for m in task_doc['metrics']
                    ],
                )
            return None

        except PyMongoError as e:
            logger.error(f"😡 Ошибка получения метрик задачи: {e}")
            return None

    def task_exists(
            self,
            task_id: str
    ) -> bool:
        """Проверка существования задачи"""
        task = self.collection.find_one({'task_id': task_id}, {'_id': 1})
        return task is not None

    def delete_task(
            self,
            task_id: str
    ) -> bool:
        """Удаление документа задачи"""
        result = self.collection.delete_one({'task_id': task_id})
        
        if result.deleted_count > 0:
            logger.debug(f"Задача {task_id} удалена")
            return True
        else:
            logger.warning(f'Задача {task_id} не найдена.')
            return False
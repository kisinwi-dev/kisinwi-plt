from typing import Optional
from pymongo.errors import PyMongoError

from .storebase import ManagerBase
from app.api.schemes import *
from app.logs import get_logger

logger = get_logger(__name__)

class CVMetricManager(ManagerBase):

    def add_response(
            self, 
            metric: TrainingMetricAdd
    ) -> bool:
        """Добавление нового ответа"""
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
    

    def add_metric(
        self, 
        metric: TrainingMetricAdd
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


    def add_metrics(
            self, 
            metrics: TrainingMetricAdds
    ) -> bool:
        """Добавление нескольких метрик для задачи"""
        try:
            # Поиск задачи
            task_doc = self.task_exists(metrics.task_id)
            
            if task_doc:
                # Задача существует
                for metric in metrics.metrics:
                    result = self.collection.update_one(
                        {
                            'task_id': metrics.task_id,
                            'metrics.metric': metric.name
                        },
                        {
                            '$push': {
                                f'metrics.$.values': {'$each': metric.values}
                            }
                        }
                    )
                    
                    # Если метрика не найдена, добавляем новую
                    if result.matched_count == 0:
                        new_metric = {
                            'metric': metric.name,
                            'values': metric.values,
                        }
                        self.collection.update_one(
                            {'task_id': metrics.task_id},
                            {'$push': {'metrics': new_metric}}
                        )
            else:
                # Новая задача - создаем документ
                new_task = {
                    'task_id': metrics.task_id,
                    'metrics': [
                        {
                            'metric': metric.name,
                            'values': metric.values,
                        }
                        for metric in metrics.metrics
                    ]
                }
                self.collection.insert_one(new_task)
            
            metric_name = ",".join(metric.name for metric in metrics.metrics)

            logger.debug(f"➕ Добавлены метрики ({metric_name}) для задачи {metrics.task_id}")
            return True
            
        except PyMongoError as e:
            logger.error(f"😡 Ошибка добавления метрик: {e}")
            return False

    def get_task_metrics(
        self, 
        task_id: str
    ) -> Optional[TaskTrainingMetrics]:
        """Получение всех метрик для задачи"""
        try:
            task_doc = self.collection.find_one({'task_id': task_id})
            
            if task_doc:
                return TaskTrainingMetrics(
                    task_id=task_doc['task_id'],
                    metrics=[
                        TrainingMetricData(
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
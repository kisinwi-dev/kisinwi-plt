from typing import Optional
from pymongo.errors import PyMongoError

from .mongo import ManagerBase
from app.api.schemes import *
from app.logs import get_logger

logger = get_logger(__name__)

class CVMetricManager(ManagerBase):

    def add_metric(
        self, 
        metric: ModelMetricAdd
    ) -> bool:
        """Добавление новой метрики для модели"""
        try:
            # Поиск модели
            model_doc = self.model_metrics_exists(metric.model_id)

            if model_doc:
                # Модель существует
                result = self.collection.update_one(
                    {
                        'model_id': metric.model_id,
                        'metrics.metric': metric.metric.name
                    },
                    {
                        '$push': {
                            f'metrics.$.values': metric.metric.values,
                        }
                    }
                )

                # Если метрика не найдена, добавляем новую
                if result.matched_count == 0:
                    new_metric = {
                        'metric': metric.metric.name,
                        'values': [metric.metric.values],
                    }
                    self.collection.update_one(
                        {'model_id': metric.model_id},
                        {
                            '$push': {'metrics': new_metric}
                        }
                    )
            else:

                # Новая модель, создаем документ
                new_task = {
                    'model_id': metric.model_id,
                    'metrics': [{
                        'metric': metric.metric.name,
                        'values': [metric.metric.values],
                    }]
                }
                self.collection.insert_one(new_task)

            logger.debug(f"Добавлена метрика {metric.metric.name}={metric.metric.values} для модели {metric.model_id}")
            return True

        except PyMongoError as e:
            logger.error(f"Ошибка добавления метрики: {e}")
            return False


    def add_metrics(
            self, 
            metrics: ModelMetricAdds
    ) -> bool:
        """Добавление нескольких метрик для модели"""
        try:
            # Поиск модели
            model_doc = self.model_metrics_exists(metrics.model_id)
            
            if model_doc:
                # Задача существует
                for metric in metrics.metrics:
                    result = self.collection.update_one(
                        {
                            'model_id': metrics.model_id,
                            'metrics.metric': metric.name
                        },
                        {
                            '$push': {
                                f'metrics.$.values': {'$each': metric.values}
                            }
                        }
                    )
                    
                    # Если метрики не найдены, сооздаём её
                    if result.matched_count == 0:
                        new_metric = {
                            'metric': metric.name,
                            'values': metric.values,
                        }
                        self.collection.update_one(
                            {'model_id': metrics.model_id},
                            {'$push': {'metrics': new_metric}}
                        )
            else:
                # Если не найдена модель создаем документ
                new_task = {
                    'model_id': metrics.model_id,
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

            logger.debug(f"Добавлены метрики ({metric_name}) для модели (id:{metrics.model_id})")
            return True
            
        except PyMongoError as e:
            logger.error(f"Ошибка добавления метрик модели(id{metrics.model_id}): {e}")
            return False

    def get_model_metrics(
        self, 
        model_id: str
    ) -> Optional[ModelMetrics]:
        """Получение всех метрик для модели"""
        try:
            model_doc = self.collection.find_one({'model_id': model_id})
            
            if model_doc:
                return ModelMetrics(
                    model_id=model_doc['model_id'],
                    metrics=[
                        ModelMetricData(
                            name=m['metric'],
                            values=m['values'],
                        )
                        for m in model_doc['metrics']
                    ],
                )
            return None

        except PyMongoError as e:
            logger.error(f"Ошибка получения метрик модели(id={model_id}): {e}")
            return None

    def model_metrics_exists(
            self,
            model_id: str
    ) -> bool:
        """Проверка существования метрик для модели"""
        task = self.collection.find_one({'model_id': model_id}, {'_id': 1})
        return task is not None

    def delete_metric(
            self,
            model_id: str
    ) -> bool:
        """Удаление документа метрик модели"""
        result = self.collection.delete_one({'model_id': model_id})
        
        if result.deleted_count > 0:
            logger.debug(f"Метрики для модели(id:{model_id}) удалены")
            return True
        else:
            logger.warning(f"Метрики модели(id:{model_id}) не найдены")
            return False
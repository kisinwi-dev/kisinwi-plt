from typing import Optional, Tuple
from pymongo.errors import PyMongoError

from .mongo import ManagerBase
from app.api.schemas import *
from app.logs import get_logger

logger = get_logger(__name__)

SPLITS = ("train", "val", "test")

def parse_split(name: str) -> Tuple[str, str]:
    """
    Определение выборки по префиксу названия метрики.

    'train_loss' -> ('train', 'loss'); название без известного
    префикса относится к тренировочной выборке.
    """
    for split in SPLITS:
        prefix = f"{split}_"
        if name.startswith(prefix):
            return split, name[len(prefix):]

    logger.warning(f"Метрика '{name}' без префикса выборки — отнесена к train")
    return "train", name

def resolve_split(metric: ModelMetricData) -> Tuple[str, str]:
    """Выборка и чистое название метрики: явное поле split приоритетнее префикса"""
    if metric.split is not None:
        return metric.split, metric.name
    return parse_split(metric.name)

class CVMetricManager(ManagerBase):

    def ensure_indexes(self):
        """Уникальный индекс по model_id: защита от дубликатов и ускорение поиска"""
        try:
            self.collection.create_index('model_id', unique=True)
        except PyMongoError as e:
            logger.error(f"Не удалось создать индекс model_id: {e}")

    def _push_metric(
        self,
        model_id: str,
        metric: ModelMetricData
    ):
        """Дозапись значений метрики в массив её выборки (метрика создаётся при отсутствии)"""
        split, name = resolve_split(metric)

        result = self.collection.update_one(
            {
                'model_id': model_id,
                f'splits.{split}.metric': name
            },
            {
                '$push': {
                    f'splits.{split}.$.values': {'$each': metric.values}
                }
            }
        )

        # Если метрика не найдена, добавляем новую
        if result.matched_count == 0:
            new_metric = {
                'metric': name,
                'values': metric.values,
            }
            self.collection.update_one(
                {'model_id': model_id},
                {'$push': {f'splits.{split}': new_metric}}
            )

    @staticmethod
    def _new_document(model_id: str, metrics: list) -> dict:
        """Документ новой модели с метриками, разнесёнными по выборкам"""
        splits = {split: [] for split in SPLITS}
        for metric in metrics:
            split, name = resolve_split(metric)
            splits[split].append({
                'metric': name,
                'values': metric.values,
            })
        return {
            'model_id': model_id,
            'splits': splits,
        }

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
                self._push_metric(metric.model_id, metric.metric)
            else:
                # Новая модель, создаем документ
                self.collection.insert_one(
                    self._new_document(metric.model_id, [metric.metric])
                )

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
                # Модель существует
                for metric in metrics.metrics:
                    self._push_metric(metrics.model_id, metric)
            else:
                # Если не найдена модель создаем документ
                self.collection.insert_one(
                    self._new_document(metrics.model_id, metrics.metrics)
                )

            metric_name = ",".join(metric.name for metric in metrics.metrics)

            logger.debug(f"Добавлены метрики ({metric_name}) для модели (id:{metrics.model_id})")
            return True

        except PyMongoError as e:
            logger.error(f"Ошибка добавления метрик модели(id:{metrics.model_id}): {e}")
            return False

    @staticmethod
    def _doc_to_model_metrics(model_doc: dict) -> ModelMetrics:
        """Сборка ответа из документа: метрики по выборкам"""
        splits = model_doc.get('splits', {})
        return ModelMetrics(
            model_id=model_doc['model_id'],
            **{
                split: [
                    ModelMetricData(
                        name=m['metric'],
                        split=split,
                        values=m['values'],
                    )
                    for m in splits.get(split, [])
                ]
                for split in SPLITS
            },
        )

    def get_model_metrics(
        self,
        model_id: str
    ) -> Optional[ModelMetrics]:
        """Получение всех метрик для модели"""
        try:
            model_doc = self.collection.find_one({'model_id': model_id})

            if model_doc:
                return self._doc_to_model_metrics(model_doc)
            return None

        except PyMongoError as e:
            logger.error(f"Ошибка получения метрик модели(id={model_id}): {e}")
            return None

    def get_models_metrics(
        self,
        model_ids: list
    ) -> list:
        """Получение метрик сразу нескольких моделей одним запросом"""
        try:
            docs = self.collection.find({'model_id': {'$in': model_ids}})
            return [self._doc_to_model_metrics(doc) for doc in docs]
        except PyMongoError as e:
            logger.error(f"Ошибка получения метрик моделей: {e}")
            return []

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

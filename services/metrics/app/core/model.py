from datetime import datetime, timezone
from typing import Optional, Tuple
from pymongo.errors import PyMongoError

from .mongo import ManagerBase
from app.api.schemas import *
from app.logs import get_logger

logger = get_logger(__name__)

SPLITS = ("train", "val", "test")

FINAL_TRAINING_STATUSES = ("completed", "failed", "cancelled")

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
        metric: ModelMetricData,
        timestamp: datetime,
    ):
        """Дозапись значений метрики и меток времени в массивы её выборки (метрика создаётся при отсутствии)"""
        split, name = resolve_split(metric)
        timestamps = [timestamp] * len(metric.values)

        result = self.collection.update_one(
            {
                'model_id': model_id,
                f'splits.{split}.metric': name
            },
            {
                '$push': {
                    f'splits.{split}.$.values': {'$each': metric.values},
                    # На старых документах без timestamps $push создаст массив сам:
                    # он окажется короче values, чтение выравнивает их с конца
                    f'splits.{split}.$.timestamps': {'$each': timestamps},
                }
            }
        )

        # Если метрика не найдена, добавляем новую
        if result.matched_count == 0:
            new_metric = {
                'metric': name,
                'values': metric.values,
                'timestamps': timestamps,
            }
            self.collection.update_one(
                {'model_id': model_id},
                {'$push': {f'splits.{split}': new_metric}}
            )

    @staticmethod
    def _new_document(model_id: str, metrics: list, timestamp: datetime) -> dict:
        """Документ новой модели с метриками, разнесёнными по выборкам"""
        splits = {split: [] for split in SPLITS}
        for metric in metrics:
            split, name = resolve_split(metric)
            splits[split].append({
                'metric': name,
                'values': metric.values,
                'timestamps': [timestamp] * len(metric.values),
            })
        return {
            'model_id': model_id,
            'status': 'in_progress',
            'splits': splits,
        }

    def add_metric(
        self,
        metric: ModelMetricAdd
    ) -> bool:
        """Добавление новой метрики для модели"""
        try:
            ts = metric.timestamp or datetime.now(timezone.utc)

            # Поиск модели
            model_doc = self.model_metrics_exists(metric.model_id)

            if model_doc:
                # Модель существует
                self._push_metric(metric.model_id, metric.metric, ts)
            else:
                # Новая модель, создаем документ
                self.collection.insert_one(
                    self._new_document(metric.model_id, [metric.metric], ts)
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
            # Одна метка времени на весь батч (все метрики эпохи записаны одновременно)
            ts = metrics.timestamp or datetime.now(timezone.utc)

            # Поиск модели
            model_doc = self.model_metrics_exists(metrics.model_id)

            if model_doc:
                # Модель существует
                for metric in metrics.metrics:
                    self._push_metric(metrics.model_id, metric, ts)
            else:
                # Если не найдена модель создаем документ
                self.collection.insert_one(
                    self._new_document(metrics.model_id, metrics.metrics, ts)
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
            status=model_doc.get('status'),
            **{
                split: [
                    ModelMetricData(
                        name=m['metric'],
                        split=split,
                        values=m['values'],
                        timestamps=m.get('timestamps', []),
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

    def set_training_status(
            self,
            model_id: str,
            status: str
    ) -> bool:
        """
        Установка статуса обучения модели.

        Upsert: если обучение упало до записи первой метрики,
        документ создаётся с пустыми выборками.
        """
        try:
            self.collection.update_one(
                {'model_id': model_id},
                {
                    '$set': {'status': status},
                    '$setOnInsert': {'splits': {split: [] for split in SPLITS}},
                },
                upsert=True,
            )
            logger.debug(f"Статус обучения модели(id:{model_id}) установлен: {status}")
            return True
        except PyMongoError as e:
            logger.error(f"Ошибка установки статуса обучения модели(id:{model_id}): {e}")
            return False

    def set_class_report(
            self,
            model_id: str,
            report: ClassReportAdd
    ) -> bool:
        """Сохранение отчёта по классам (идемпотентная перезапись; upsert для модели без метрик)"""
        try:
            self.collection.update_one(
                {'model_id': model_id},
                {
                    '$set': {'class_report': report.model_dump()},
                    '$setOnInsert': {'splits': {split: [] for split in SPLITS}},
                },
                upsert=True,
            )
            logger.debug(f"Class report модели(id:{model_id}) сохранён")
            return True
        except PyMongoError as e:
            logger.error(f"Ошибка сохранения class report модели(id:{model_id}): {e}")
            return False

    def get_class_report(
            self,
            model_id: str
    ) -> Optional[ClassReport]:
        """Получение отчёта по классам модели"""
        try:
            doc = self.collection.find_one(
                {'model_id': model_id},
                {'class_report': 1}
            )
            if doc is None or 'class_report' not in doc:
                return None
            return ClassReport(model_id=model_id, **doc['class_report'])
        except PyMongoError as e:
            logger.error(f"Ошибка получения class report модели(id:{model_id}): {e}")
            return None

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

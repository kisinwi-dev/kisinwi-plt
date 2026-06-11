"""
Одноразовая миграция коллекции training_cv на формат с разбиением по выборкам.

Старый формат: {'model_id': ..., 'metrics': [{'metric': 'train_loss', 'values': [...]}]}
Новый формат:  {'model_id': ..., 'splits': {'train': [{'metric': 'loss', 'values': [...]}], 'val': [...], 'test': [...]}}

Запуск из корня сервиса: uv run python -m scripts.migrate_to_splits
Идемпотентен: документы без старого поля 'metrics' пропускаются.
"""
from pymongo import MongoClient

from app.config import mongodb_config
from app.core.model import SPLITS, parse_split
from app.logs import get_logger

logger = get_logger(__name__)


def migrate() -> None:
    client = MongoClient(mongodb_config.URL_METRIC)
    try:
        collection = client[mongodb_config.DATABASE_METRIC][mongodb_config.COLLECTION_TRAINING_CV]

        migrated = 0
        skipped = 0
        for doc in collection.find({}):
            if 'metrics' not in doc:
                skipped += 1
                continue

            splits = doc.get('splits') or {split: [] for split in SPLITS}
            for old_metric in doc['metrics']:
                split, name = parse_split(old_metric['metric'])
                splits[split].append({
                    'metric': name,
                    'values': old_metric['values'],
                })

            collection.update_one(
                {'_id': doc['_id']},
                {
                    '$set': {'splits': splits},
                    '$unset': {'metrics': ''},
                }
            )
            migrated += 1
            logger.info(f"Мигрирован документ модели (id:{doc.get('model_id')})")

        logger.info(f"Готово: мигрировано {migrated}, пропущено {skipped}")
    finally:
        client.close()


if __name__ == "__main__":
    migrate()

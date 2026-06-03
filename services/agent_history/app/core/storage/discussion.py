import json
import shutil
from typing import List
from pydantic import ValidationError

from app.logs import get_logger
from .base import BaseStorage

logger = get_logger(__name__)


class DiscussionStorage(BaseStorage):
    def get_history(self, discussion_id: str) -> List[dict] | None:
        discussion_dir = self.base_path / discussion_id

        if not discussion_dir.exists():
            return None

        events = []
        for pattern in ["system/*.json", "responses/*.json", "tools/**/*.json"]:
            for filepath in discussion_dir.glob(pattern):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        events.append(json.load(f))
                except (json.JSONDecodeError, ValidationError, KeyError) as e:
                    logger.error(f"Ошибка при чтении файла {filepath}: {e}")
                    continue

        events.sort(key=lambda x: x["timestamp"])
        return events

    def get_all(self) -> List[str]:
        return [d.name for d in self.base_path.iterdir() if d.is_dir()]

    def delete(self, discussion_id: str) -> bool:
        discussion_dir = self.base_path / discussion_id

        if discussion_dir.exists():
            shutil.rmtree(discussion_dir)
            return True
        return False

from uuid import uuid4

from app.api.schemas import SystemMessage
from app.logs import get_logger
from .base import BaseStorage

logger = get_logger(__name__)


class SystemStorage(BaseStorage):
    def save(self, discussion_id: str, message: SystemMessage) -> str:
        system_dir = self.base_path / discussion_id / "system"
        system_dir.mkdir(parents=True, exist_ok=True)

        filepath = system_dir / f"{uuid4()}.json"

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(message.model_dump_json(indent=2, ensure_ascii=False))

        return str(filepath)

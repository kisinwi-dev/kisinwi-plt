import json
import aiofiles
from pydantic import ValidationError
from typing import List
from uuid import uuid4

from app.api.schemas import SystemMessage
from app.logs import get_logger
from .base import BaseStorage

logger = get_logger(__name__)


class SystemStorage(BaseStorage):
    async def save(self, discussion_id: str, message: SystemMessage) -> str:
        system_dir = self.base_path / discussion_id / "system"
        system_dir.mkdir(parents=True, exist_ok=True)

        filepath = system_dir / f"{uuid4()}.json"

        async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
            await f.write(message.model_dump_json(indent=2, ensure_ascii=False))

        return str(filepath)

    async def get_all(self, discussion_id: str) -> List[SystemMessage] | None:
        discussion_dir = self.base_path / discussion_id

        if not discussion_dir.exists():
            return None

        system_dir = discussion_dir / "system"

        if not system_dir.exists():
            return []

        messages = []
        for filepath in system_dir.glob("*.json"):
            try:
                async with aiofiles.open(filepath, 'r', encoding='utf-8') as f:
                    messages.append(SystemMessage(**json.loads(await f.read())))
            except (json.JSONDecodeError, ValidationError, KeyError) as e:
                logger.error(f"Ошибка при чтении {filepath}: {e}")

        messages.sort(key=lambda x: x.timestamp)
        return messages

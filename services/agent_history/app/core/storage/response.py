import json
import aiofiles
from typing import Optional
from pydantic import ValidationError

from app.api.schemas import AgentResponse
from app.logs import get_logger
from .base import BaseStorage

logger = get_logger(__name__)


class ResponseStorage(BaseStorage):
    async def save(self, discussion_id: str, response: AgentResponse) -> str:
        responses_dir = self.base_path / discussion_id / "responses"
        responses_dir.mkdir(parents=True, exist_ok=True)

        filepath = responses_dir / f"{response.response_id}.json"

        async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
            await f.write(response.model_dump_json(indent=2, ensure_ascii=False))

        return str(filepath)

    async def get(self, discussion_id: str, response_id: str) -> Optional[AgentResponse]:
        discussion_dir = self.base_path / discussion_id

        if not discussion_dir.exists():
            return None

        filepath = discussion_dir / "responses" / f"{response_id}.json"

        try:
            async with aiofiles.open(filepath, 'r', encoding='utf-8') as f:
                return AgentResponse(**json.loads(await f.read()))
        except (json.JSONDecodeError, ValidationError, KeyError) as e:
            logger.error(f"Ошибка при чтении файла {filepath}: {e}")
            raise e

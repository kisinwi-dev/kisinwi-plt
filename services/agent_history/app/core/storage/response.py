import json
import aiofiles
from pydantic import ValidationError
from typing import List

from app.api.schemas import AgentResponse, AgentStatus
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

    async def cancel_in_progress(self, discussion_id: str) -> int:
        """
        Пометить все ещё выполняющиеся (IN_PROGRESS) ответы как CANCELLED.

        Нужно при остановке пайплайна: процесс агентов убит снаружи и сам не
        успел финализировать текущего агента, иначе он навсегда завис бы
        IN_PROGRESS. Возвращает число помеченных ответов.
        """
        responses = await self.get_all(discussion_id)
        if not responses:
            return 0
        count = 0
        for response in responses:
            if response.status == AgentStatus.IN_PROGRESS:
                response.status = AgentStatus.CANCELLED
                await self.save(discussion_id, response)
                count += 1
        return count

    async def get_all(self, discussion_id: str) -> List[AgentResponse] | None:
        discussion_dir = self.base_path / discussion_id

        if not discussion_dir.exists():
            return None

        responses_dir = discussion_dir / "responses"

        if not responses_dir.exists():
            return []

        responses = []
        for filepath in responses_dir.glob("*.json"):
            try:
                async with aiofiles.open(filepath, 'r', encoding='utf-8') as f:
                    responses.append(AgentResponse(**json.loads(await f.read())))
            except (json.JSONDecodeError, ValidationError, KeyError) as e:
                logger.error(f"Ошибка при чтении {filepath}: {e}")

        responses.sort(key=lambda x: x.timestamp)
        return responses

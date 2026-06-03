import aiofiles

from app.api.schemas import AgentResponse
from .base import BaseStorage


class ResponseStorage(BaseStorage):
    async def save(self, discussion_id: str, response: AgentResponse) -> str:
        responses_dir = self.base_path / discussion_id / "responses"
        responses_dir.mkdir(parents=True, exist_ok=True)

        filepath = responses_dir / f"{response.response_id}.json"

        async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
            await f.write(response.model_dump_json(indent=2, ensure_ascii=False))

        return str(filepath)

import json
import aiofiles
from typing import List

from app.api.schemas import Tool
from app.logs import get_logger
from .base import BaseStorage

logger = get_logger(__name__)


class ToolStorage(BaseStorage):
    async def save(self, discussion_id: str, tool: Tool) -> str:
        response_id = tool.response_id or "_unlinked"
        tool_dir = self.base_path / discussion_id / "tools" / response_id
        tool_dir.mkdir(parents=True, exist_ok=True)

        filepath = tool_dir / f"{tool.id}.json"

        async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
            await f.write(tool.model_dump_json(indent=2, ensure_ascii=False))

        return str(filepath)

    async def update(self, discussion_id: str, tool: Tool) -> str:
        return await self.save(discussion_id, tool)

    async def get_by_response(self, discussion_id: str, response_id: str) -> List[dict] | None:
        discussion_dir = self.base_path / discussion_id

        if not discussion_dir.exists():
            return None

        tools_dir = discussion_dir / "tools" / response_id

        if not tools_dir.exists():
            return []

        tools = []
        for filepath in tools_dir.glob("*.json"):
            try:
                async with aiofiles.open(filepath, 'r', encoding='utf-8') as f:
                    tools.append(json.loads(await f.read()))
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Ошибка при чтении файла {filepath}: {e}")
                continue

        tools.sort(key=lambda x: x.get("timestamp", ""))
        return tools

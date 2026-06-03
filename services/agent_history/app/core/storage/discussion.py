import json
import shutil
import aiofiles
from datetime import datetime
from typing import List, Optional
from uuid import uuid4
from pydantic import ValidationError

from app.api.schemas import DiscussionMeta, DiscussionMetaUpdate, CreateDiscussion
from app.logs import get_logger
from .base import BaseStorage

logger = get_logger(__name__)

_META_FILE = "meta.json"


class DiscussionStorage(BaseStorage):

    async def create(self, data: CreateDiscussion) -> DiscussionMeta:
        discussion_id = data.discussion_id or str(uuid4())
        discussion_dir = self.base_path / discussion_id
        discussion_dir.mkdir(parents=True, exist_ok=True)

        meta = DiscussionMeta(
            discussion_id=discussion_id,
            title=data.title,
            tags=data.tags,
            pipeline=data.pipeline,
        )

        await self._write_meta(discussion_id, meta)
        return meta

    async def get_meta(self, discussion_id: str) -> Optional[DiscussionMeta]:
        meta_path = self.base_path / discussion_id / _META_FILE

        if not meta_path.exists():
            return None

        try:
            async with aiofiles.open(meta_path, 'r', encoding='utf-8') as f:
                return DiscussionMeta(**json.loads(await f.read()))
        except (json.JSONDecodeError, ValidationError, KeyError) as e:
            logger.error(f"Ошибка при чтении {meta_path}: {e}")
            return None

    async def update_meta(self, discussion_id: str, update: DiscussionMetaUpdate) -> Optional[DiscussionMeta]:
        meta = await self.get_meta(discussion_id)

        if meta is None:
            return None

        if update.title is not None:
            meta.title = update.title
        if update.status is not None:
            meta.status = update.status
        if update.tags is not None:
            meta.tags = update.tags
        if update.pipeline is not None:
            meta.pipeline = update.pipeline

        meta.updated_at = datetime.now()
        await self._write_meta(discussion_id, meta)
        return meta

    async def get_history(self, discussion_id: str) -> List[dict] | None:
        discussion_dir = self.base_path / discussion_id

        if not discussion_dir.exists():
            return None

        events = []
        for pattern in ["system/*.json", "responses/*.json", "tools/**/*.json"]:
            for filepath in discussion_dir.glob(pattern):
                try:
                    async with aiofiles.open(filepath, 'r', encoding='utf-8') as f:
                        events.append(json.loads(await f.read()))
                except (json.JSONDecodeError, ValidationError, KeyError) as e:
                    logger.error(f"Ошибка при чтении файла {filepath}: {e}")
                    continue

        events.sort(key=lambda x: x["timestamp"])
        return events

    async def get_all(self) -> List[DiscussionMeta]:
        result = []
        for d in self.base_path.iterdir():
            if not d.is_dir():
                continue
            meta = await self.get_meta(d.name)
            if meta is None:
                meta = DiscussionMeta(discussion_id=d.name)
            result.append(meta)
        return result

    def delete(self, discussion_id: str) -> bool:
        discussion_dir = self.base_path / discussion_id

        if discussion_dir.exists():
            shutil.rmtree(discussion_dir)
            return True
        return False

    async def _write_meta(self, discussion_id: str, meta: DiscussionMeta) -> None:
        meta_path = self.base_path / discussion_id / _META_FILE
        async with aiofiles.open(meta_path, 'w', encoding='utf-8') as f:
            await f.write(meta.model_dump_json(indent=2, ensure_ascii=False))

import asyncio
import json
import shutil
import aiofiles
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from uuid import uuid4
from pydantic import ValidationError

from app.api.schemas import DiscussionMeta, DiscussionMetaUpdate, CreateDiscussion, DiscussionStatus
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
            agent_roles=data.agent_roles,
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
        if update.agent_roles is not None:
            meta.agent_roles = update.agent_roles

        if meta.finished_at is None and meta.status in (DiscussionStatus.COMPLETED, DiscussionStatus.FAILED):
            meta.finished_at = datetime.now()
        await self._write_meta(discussion_id, meta)
        return meta

    async def get_history(self, discussion_id: str) -> List[dict] | None:
        discussion_dir = self.base_path / discussion_id

        if not discussion_dir.exists():
            return None

        filepaths: list[Path] = []
        for pattern in ["system/*.json", "responses/*.json", "tools/**/*.json"]:
            filepaths.extend(discussion_dir.glob(pattern))

        async def _read(filepath: Path) -> dict | None:
            try:
                async with aiofiles.open(filepath, 'r', encoding='utf-8') as f:
                    return json.loads(await f.read())
            except (json.JSONDecodeError, ValidationError, KeyError) as e:
                logger.error(f"Ошибка при чтении файла {filepath}: {e}")
                return None

        results = await asyncio.gather(*[_read(fp) for fp in filepaths])
        events = [r for r in results if r is not None]
        events.sort(key=lambda x: x.get("timestamp", ""))
        return events

    async def get_all(
        self,
        status: Optional[DiscussionStatus] = None,
        pipeline: Optional[str] = None,
        skip: int = 0,
        limit: int = 50,
    ) -> List[DiscussionMeta]:
        result = []
        for d in self.base_path.iterdir():
            if not d.is_dir():
                continue
            meta = await self.get_meta(d.name)
            if meta is None:
                meta = DiscussionMeta(discussion_id=d.name)
            if status is not None and meta.status != status:
                continue
            if pipeline is not None and meta.pipeline != pipeline:
                continue
            result.append(meta)
        result.sort(key=lambda x: x.created_at, reverse=True)
        return result[skip : skip + limit]

    async def delete(self, discussion_id: str) -> bool:
        discussion_dir = self.base_path / discussion_id

        if discussion_dir.exists():
            await asyncio.to_thread(shutil.rmtree, discussion_dir)
            return True
        return False

    async def _write_meta(self, discussion_id: str, meta: DiscussionMeta) -> None:
        meta_path = self.base_path / discussion_id / _META_FILE
        async with aiofiles.open(meta_path, 'w', encoding='utf-8') as f:
            await f.write(meta.model_dump_json(indent=2, ensure_ascii=False))

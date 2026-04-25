import asyncio
from typing import Dict

tasks_db: Dict[str, dict] = {}

task_queue: asyncio.Queue = asyncio.Queue()
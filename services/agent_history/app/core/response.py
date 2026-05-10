import json
import shutil
from pathlib import Path
from typing import List, Optional
from pydantic import ValidationError

from app.api.schemas import AgentResponse
from app.logs import get_logger

logger = get_logger(__name__)

class AgentResponseStorage:
    """
    Хранилище ответов агентов в JSON-файлах.

    Структура:
        discussion/
          ├── {discussion_id}/
          │   ├── {response_id}.json
          │   └── ...
          └── ...
    """
    
    def __init__(self, base_path: str = "discussion"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
    
    def save_response(
            self, 
            discussion_id: str, 
            response: AgentResponse
    ):
        """Сохранить ответ агента в JSON файл"""

        # апка для дискуссий
        discussion_dir = self.base_path / discussion_id
        discussion_dir.mkdir(exist_ok=True)
        
        # имя файла
        filename = f"{response.response_id}.json"
        filepath = discussion_dir / filename

        # Сохраняем JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(response.model_dump(), f, ensure_ascii=False, indent=2)

        return str(filepath)
    
    def get_discussion_history(self, discussion_id: str) -> List[AgentResponse]:
        """Получить всю историю дискуссии по ID (Автоматическая сортировка по времени)"""
        discussion_dir = self.base_path / discussion_id

        if not discussion_dir.exists():
            return []

        responses = []
        for filepath in sorted(discussion_dir.glob("*.json")):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    response = json.load(f)
                    responses.append(AgentResponse(**response))
            except (json.JSONDecodeError, ValidationError, KeyError) as e:
                logger.error(f"Ошибка при чтении файла {filepath}: {e}")
                continue 

        responses.sort(key=lambda x: x.timestamp)

        return responses

    def get_agent_responses(self, discussion_id: str, response_id: str) -> Optional[AgentResponse]:
        """Получить конкретный ответ в дискуссии"""
        discussion_dir = self.base_path / discussion_id

        if not discussion_dir.exists():
            return None

        filepath = discussion_dir / f"{response_id}.json"

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                response = json.load(f)
                response = AgentResponse(**response)
        except (json.JSONDecodeError, ValidationError, KeyError) as e:
            logger.error(f"Ошибка при чтении файла {filepath}: {e}")
            raise e

        return response

    def get_all_discussions(self) -> List[str]:
        """Получить список всех ID дискуссий"""
        return [d.name for d in self.base_path.iterdir() if d.is_dir()]

    def delete_discussion(self, discussion_id: str) -> bool:
        """Удалить всю дискуссию (папку с файлами)"""
        discussion_dir = self.base_path / discussion_id

        if discussion_dir.exists():
            shutil.rmtree(discussion_dir)
            return True
        return False

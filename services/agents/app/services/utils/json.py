import json
import re
from typing import Dict

from app.logs import get_logger

logger = get_logger(__name__)

def parse_in_json(
    data: Dict | str
) -> Dict:
    """Парсинг в JSON из строки или возврат словаря"""
    if isinstance(data, dict):
        return data
    
    # Очищаем строку от маркеров markdown
    data = re.sub(r'```json\s*\n?', '', data)
    cleaned = re.sub(r'```\s*\n?', '', data)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # LLM иногда отдаёт дважды экранированную JSON-строку (литеральные \n и \").
    # Снимаем один слой экранирования, декодировав её как JSON-строку, и парсим повторно.
    try:
        unescaped = json.loads(f'"{cleaned}"')
        return json.loads(unescaped)
    except json.JSONDecodeError as e:
        logger.error(f"Ошибка парсинга JSON. \nПолученный текст:\n{cleaned}\nОшибка: {e}")
        raise
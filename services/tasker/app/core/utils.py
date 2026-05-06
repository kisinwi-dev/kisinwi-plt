import uuid

def valid_uuid(
        val: str, 
        on_error: bool = False
    ) -> bool:
    """
    Проверяет, является ли строка валидным UUID

    Args:
        val: Проверяемая строка
        on_error: Если включить выбрасывает ошибку. По умолчанию выключен.

    Raises:
        ValueError: Если raise_on_error=True и значение невалидный UUID
    """
    try:
        uuid.UUID(str(val))
        return True
    except (ValueError, AttributeError, TypeError):
        if on_error:
            raise ValueError(f"Некорректный формат UUID: {val}")
        return False
from typing import Any

class CoreException(Exception):
    """
    Базовый класс для всех ошибок.
    """
    def __init__(
            self, 
            message: str, 
            detail: str | dict[str, Any] | None = None,
            status_code: int = 400,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail
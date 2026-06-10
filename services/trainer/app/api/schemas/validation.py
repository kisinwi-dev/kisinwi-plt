from typing import List
from pydantic import BaseModel, Field

class ConfigValidationError(BaseModel):
    loc: str = Field(..., description="Путь к полю конфигурации (например, 'trainer_params.optimizer.name')")
    msg: str = Field(..., description="Описание ошибки")

class ConfigValidationResponse(BaseModel):
    valid: bool = Field(..., description="Валидна ли конфигурация")
    errors: List[ConfigValidationError] = Field(
        default_factory=list,
        description="Список ошибок валидации (пустой, если конфигурация валидна)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "valid": False,
                "errors": [
                    {
                        "loc": "trainer_params.optimizer.name",
                        "msg": "Оптимизатор 'Adamm' не найден среди доступных"
                    }
                ]
            }
        }
    }

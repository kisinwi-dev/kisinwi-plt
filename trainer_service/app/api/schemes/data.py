from pydantic import BaseModel, Field

class DataLoaderParams(BaseModel):
    dataset_id: str = Field(..., description="Id датасета")
    version_id: str = Field(..., description="Версия датасета")
    img_w_size: int = Field(..., description="Ширина изображений")
    img_h_size: int = Field(..., description="Высота изображений")
    batch_size: int = Field(default=32, description="Количество изображений в батче")
    is_calculate_normalize_dataset: bool = Field(default=False, description="Рассчитать нормализацию для датасета")

    model_config = {
        "json_schema_extra": {
            "example": {
                "dataset_id": "dataset_123",
                "version_id": "v1.0",
                "img_w_size": 224,
                "img_h_size": 224,
                "batch_size": 32,
                "is_calculate_normalize_dataset": False
            }
        }
    }
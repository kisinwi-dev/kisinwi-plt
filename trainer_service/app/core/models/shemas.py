from pydantic import BaseModel, Field

class ModelParams(BaseModel):
    type: str = Field(..., description="Имя модели из timm")
    pretrained: bool = Field(False, description="Использовать предобученные веса")
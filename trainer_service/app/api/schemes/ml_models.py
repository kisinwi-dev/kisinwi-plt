from pydantic import BaseModel, Field

class ModelParams(BaseModel):
    type: str = Field(..., description="Имя модели из timm")
    pretrained: bool = Field(False, description="Использовать предобученные веса")

    model_config = {
        "json_schema_extra": {
            "example": {
                "type": "resnet50",
                "pretrained": True
            }
        }
    }
from pydantic import BaseModel 

class ModelCreate(BaseModel):
    name: str
    model_type: str
    description: str
    classes: list
    train_params: dict

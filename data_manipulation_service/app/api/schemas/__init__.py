from pydantic import BaseModel, Field

class MessageResponse(BaseModel):
    message: str = Field(
        ..., 
        example="Dataset 'dataset_1' successfully created", 
        description="General message response for actions"
    )
from pydantic import BaseModel

from .version import VersionSchema
from .class_ import ClassSchema
from .dataset import DatasetSchema

__all__ = [
    'VersionSchema', 
    'ClassSchema', 
    'DatasetSchema'
]

class MessageResponse(BaseModel):
    message: str
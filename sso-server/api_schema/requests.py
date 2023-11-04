from typing import Optional
from pydantic import BaseModel

class Vector(BaseModel):
    vector: list[str]

class ProfileRequest(BaseModel):
    id: Optional[str] = ''
    user_name: str
    name: str
    face_embedding: Optional[Vector] = None
    
    
from typing import Optional
from pydantic import BaseModel, conlist

class Vector(BaseModel):
    vector: conlist(float, min_length=1024, max_length=1024)


class ProfileRequest(BaseModel):
    id: Optional[str] = ''
    user_name: str
    name: str
    face_embedding: Vector
    
class AuthRequest(BaseModel):
    face_embedding: Vector
    redirect_uri: str
    
    
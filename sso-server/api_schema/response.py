import uuid
from pydantic import BaseModel

class ProfileResponse(BaseModel):
    id: uuid.UUID
    name: str
    user_name: str
    
    class Config():
        orm_mode = True
        from_attributes=True
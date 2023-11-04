import uuid
from pydantic import BaseModel
import datetime

class ProfileResponse(BaseModel):
    id: uuid.UUID
    name: str
    user_name: str
    
    class Config():
        orm_mode = True
        from_attributes=True
        
class ClaimResponse(BaseModel):
    token: str
    expiry: datetime.datetime
    type: str
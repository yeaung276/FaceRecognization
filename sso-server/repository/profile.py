import uuid
from typing import List

from fastapi import Depends

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from repository.models import Profile
from repository.db import get_db
from api_schema.requests import ProfileRequest
from api_schema.response import ProfileResponse
from repository.exception import UserNameError

class ProfileRepository:
    def __init__(self, db: Session= Depends(get_db)):
        self.db = db

    def insert(self, profile: ProfileRequest) -> ProfileResponse:
        try:
            new_profile = Profile(
                id=uuid.uuid4(),
                name=profile.name,
                user_name=profile.user_name
            )
            self.db.add(new_profile)
            self.db.commit()
            self.db.refresh(new_profile)
            return ProfileResponse.from_orm(new_profile)
        except IntegrityError:
            raise UserNameError()
    
    def get(self, id: uuid.UUID) -> ProfileResponse:
        profile = self.db.query(Profile).get(id)
        return ProfileResponse.from_orm(profile)
    
    def get_by_username(self, username: str) -> List[ProfileResponse]:
        profiles = self.db.query(Profile).filter(Profile.user_name==username).all()
        return [ProfileResponse.from_orm(p) for p in profiles]
    
    def exist(self, username: str) -> bool:
        return self.db.query(Profile).filter(Profile.user_name==username).first() is not None
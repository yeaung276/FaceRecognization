import uuid
from sqlalchemy.orm import Session

from repository.models import Profile
from api_schema.requests import ProfileRequest
from api_schema.response import ProfileResponse

class ProfileRepository:
    def insert(db: Session, profile: ProfileRequest) -> ProfileResponse:
        new_profile = Profile(
            id=uuid.uuid4(),
            name=profile.name,
            user_name=profile.user_name
        )
        db.add(new_profile)
        db.commit()
        db.refresh(new_profile)
        return ProfileResponse.from_orm(new_profile)
    
    def get(db: Session, id: uuid.UUID) -> ProfileResponse:
        profile = db.query(Profile).get(id)
        return ProfileResponse.from_orm(profile)
from sqlalchemy import String, Column, UUID

from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Profile(Base):
    __tablename__ = 'Profile'
    
    id =  Column(UUID, index=True, primary_key=True)
    name = Column(String)
    user_name = Column(String, unique=True)
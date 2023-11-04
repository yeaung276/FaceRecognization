from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import env

engine = create_engine(env.SQLALCHEMY_DATABASE_URL)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
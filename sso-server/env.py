import os

PORT = int(os.getenv('PORT', 8080))
SQLALCHEMY_DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql+psycopg2://admin:password@localhost:5432')
REDIS_DATABASE_URL = os.environ.get('REDIS_URL', 'localhost')
REDIS_DATABASE_PORT = int(os.environ.get('REDIS_PORT', 6379))
JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY', 'JWTSECRETKEY')
JWT_ALGORITHM = os.environ.get('JWT_ALGORITHM', 'HS256')

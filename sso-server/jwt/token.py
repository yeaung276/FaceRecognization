from datetime import datetime, timedelta
from jose import jwt

import env

JWT_EXPIRY_MINUTE = 30

def create_token(data: dict):
    expires_delta = timedelta(minutes=int(JWT_EXPIRY_MINUTE))
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({'exp': expire})
    to_encode.update({'id': str(data['id'])})
    token = jwt.encode(to_encode, env.JWT_SECRET_KEY, algorithm=env.JWT_ALGORITHM)
    return token, expire
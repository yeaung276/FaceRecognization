import uuid
from redis import Redis
import env

CODE_TTL_SECOND = 60

def get_redis():
    return Redis(env.REDIS_DATABASE_URL, env.REDIS_DATABASE_PORT)

class CodeGen:
    def create_one_time_code(redis: Redis, profile_id: uuid.UUID) -> str:
        code = uuid.uuid1()
        redis.set(str(code), profile_id.bytes, ex=CODE_TTL_SECOND)
        print(redis.keys())
        return code
    
    def claim_one_time_code(redis: Redis, code: str) -> uuid.UUID:
        response = redis.get(code)
        print(response)
        if response is None:
            raise KeyError()
        redis.delete(code)
        return uuid.UUID(bytes=response)
        
import uuid
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from fastapi.middleware.cors import CORSMiddleware

from api_schema.requests import ProfileRequest, AuthRequest
from api_schema.response import ClaimResponse
from repository.db import get_db
from repository.profile import ProfileRepository
from repository.exception import UserNameError
from redis_codegen.codegen import get_redis, CodeGen
from jwt.token import create_token

ssoApp = FastAPI()

ssoApp.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@ssoApp.on_event('startup')
async def setup():
    ssoApp.state.redis = get_redis()
    
@ssoApp.on_event('shutdown')
async def shutdown():
    ssoApp.state.redis.close()
    
@ssoApp.get('/verify')
def verify_token():
    return 'to be implemented'

@ssoApp.get('/claim')
def get_token(code: str, db:Session=Depends(get_db)):
    try:
        profile_id = CodeGen.claim_one_time_code(ssoApp.state.redis, code)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Code expired or not exist")
    profile = ProfileRepository.get(db, profile_id)
    token, expiry = create_token(profile.dict())
    return ClaimResponse(token=token, expiry=expiry, type='barer')

@ssoApp.post('/authenticate')
async def authenticate(request: AuthRequest, db:Session=Depends(get_db)):
    profiles = ProfileRepository.get_by_username(db, username=request.user_name)
    if len(profiles) == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Credentail errors")
    # here goes authenticcation by neural net
    # end
    code = CodeGen.create_one_time_code(ssoApp.state.redis, profiles[0].id)
    return RedirectResponse(url=f'{request.redirect_uri}?code={code}')
        
        

@ssoApp.post('/register')
async def register(request: ProfileRequest, db:Session=Depends(get_db)):
    try:
        return ProfileRepository.insert(db, request)
    except UserNameError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="username already taken")
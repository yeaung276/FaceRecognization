from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from api_schema.requests import ProfileRequest, AuthRequest
from api_schema.response import ClaimResponse
from repository.profile import ProfileRepository
from repository.milvus import setup as vector_setup
from redis_codegen.codegen import get_redis, CodeGen
from jwt.token import create_token
from encoder.background_service import BackgroundService
from encoder.query_service import VectorQuery
import env

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
    if env.MODE == 'setup':
        vector_setup()
    
    
@ssoApp.on_event('shutdown')
async def shutdown():
    ssoApp.state.redis.close()
    
@ssoApp.get('/verify')
def verify_token():
    return 'to be implemented'

@ssoApp.get('/claim')
def get_token(code: str, profile_repo: ProfileRepository = Depends(ProfileRepository)):
    try:
        profile_id = CodeGen.claim_one_time_code(ssoApp.state.redis, code)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Code expired or not exist")
    profile = profile_repo.get(profile_id)
    token, expiry = create_token(profile.dict())
    return ClaimResponse(token=token, expiry=expiry, type='barrer')

@ssoApp.post('/authenticate')
def authenticate(request: AuthRequest, 
                       profile_repo: ProfileRepository=Depends(ProfileRepository),
                       query_service: VectorQuery=Depends(VectorQuery)):
    user_id = query_service.authenticate(request.face_embedding)
    if user_id is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    profile = profile_repo.get(user_id)
    if profile is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    code = CodeGen.create_one_time_code(ssoApp.state.redis, profile.id)
    return RedirectResponse(url=f'{request.redirect_uri}?code={code}', status_code=status.HTTP_302_FOUND)
        
        

@ssoApp.post('/register')
async def register(request: ProfileRequest, 
                   background: BackgroundService = Depends(BackgroundService), 
                   profile_repo: ProfileRepository = Depends(ProfileRepository)):
    if profile_repo.exist(request.user_name):
        profile = profile_repo.get_by_username(username=request.user_name)[0]
    else:
        profile = profile_repo.insert(request)
    background.start_add_vector_task(request.face_embedding, profile.id)
    return profile
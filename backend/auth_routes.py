import os
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import jwt
from dotenv import load_dotenv

load_dotenv()

router = APIRouter(prefix="/auth", tags=["auth"])

# Load admin credentials from .env
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"

class LoginRequest(BaseModel):
    username: str
    password: str

@router.post("/login")
async def login(request: LoginRequest):
    """Admin login with JWT token generation"""
    if request.username != ADMIN_USERNAME or request.password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Generate JWT token with 24hr expiry
    expiration = datetime.utcnow() + timedelta(hours=24)
    token = jwt.encode(
        {"sub": request.username, "exp": expiration},
        SECRET_KEY,
        algorithm=ALGORITHM
    )
    
    return {
        "token": token,
        "username": request.username,
        "expires_at": expiration.isoformat()
    }

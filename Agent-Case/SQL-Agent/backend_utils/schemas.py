from datetime import datetime
from pydantic import BaseModel, EmailStr
from typing import Optional


class UserCreate(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: str
    email: EmailStr
    is_active: bool
    is_admin: bool
    created_at: Optional[datetime]

    class Config:
        orm_mode = True

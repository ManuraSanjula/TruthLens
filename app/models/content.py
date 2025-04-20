from datetime import datetime
from enum import Enum
from typing import Optional, Dict
from pydantic import BaseModel, Field, HttpUrl
from bson import ObjectId
from app.utils.database import db
from app.utils.config import settings


class ContentType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    URL = "url"


class ContentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ContentBase(BaseModel):
    user_id: str
    content_type: ContentType
    source_url: Optional[HttpUrl] = None
    raw_text: Optional[str] = None
    file_path: Optional[str] = None
    metadata: Dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: ContentStatus = ContentStatus.PENDING


class ContentCreate(ContentBase):
    pass


class Content(ContentBase):
    id: str = Field(..., alias="_id")

    class Config:
        allow_population_by_field_name = True
        json_encoders = {ObjectId: str}

    @classmethod
    async def create(cls, content_data: ContentCreate) -> 'Content':
        """Create new content in database"""
        content_dict = content_data.dict()
        result = await db.client[settings.MONGODB_NAME].contents.insert_one(content_dict)
        created_content = await db.client[settings.MONGODB_NAME].contents.find_one({"_id": result.inserted_id})
        return cls(**created_content)

    @classmethod
    async def get(cls, content_id: str) -> Optional['Content']:
        """Get content by ID"""
        content = await db.client[settings.MONGODB_NAME].contents.find_one({"_id": ObjectId(content_id)})
        return cls(**content) if content else None

    @classmethod
    async def update_status(cls, content_id: str, status: ContentStatus) -> bool:
        """Update content status"""
        result = await db.client[settings.MONGODB_NAME].contents.update_one(
            {"_id": ObjectId(content_id)},
            {"$set": {"status": status}}
        )
        return result.modified_count > 0
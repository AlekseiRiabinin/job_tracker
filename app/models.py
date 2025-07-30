import os
import time
from datetime import datetime
from typing import Optional, Self, ClassVar, Any
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pymongo import MongoClient
from pymongo.results import InsertOneResult, UpdateResult, DeleteResult
from pymongo.errors import ConnectionFailure, OperationFailure
from bson import ObjectId


class MongoDBConnection:
    """Connector to MongoDB."""
    _client = None
    _db = None

    @classmethod
    def get_db(cls: Self) -> Self:
        """Get MongoDB."""
        if cls._db is None:
            cls._client = cls._get_mongo_client()
            cls._db = cls._client.jobtracker
        return cls._db

    @classmethod
    def _get_mongo_client(cls: Self) -> MongoClient:
        """Establish connection to MongoDB with retry logic."""
        max_retries = 5
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                client = MongoClient(
                    f"mongodb://admin:{os.getenv('MONGO_ROOT_PASSWORD')}@mongo:27017/",
                    serverSelectionTimeoutMS=5000,
                    socketTimeoutMS=30000,
                    connectTimeoutMS=30000,
                    authSource='admin'
                )
                client.admin.command('ping')
                return client
            except (ConnectionFailure, OperationFailure) as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(retry_delay)
                continue


class JobApplicationBase(BaseModel):
    """Base schema for job application data validation."""
    VALID_STATUSES: ClassVar[set[str]] = {
        "Applied",
        "Interview",
        "Offer",
        "Rejected",
        "Ghosted"
    }

    company: str = Field(
        ..., min_length=1, max_length=100, 
        description="Name of the company applying to"
    )
    location: str = Field(
        ..., min_length=1, max_length=100,
        description="Physical location of the job"
    )
    role: str = Field(
        ..., min_length=1, max_length=100,
        description="Job title/position being applied for"
    )
    status: str = Field(
        "Applied", min_length=1, max_length=50,
        description="Current application status"
    )
    notes: Optional[str] = Field(
        None, max_length=500,
        description="Additional notes about the application"
    )
    applied_date: datetime = Field(
        default_factory=datetime.now,
        description="When the application was submitted"
    )

    @field_validator('status')
    def validate_status(cls: Self, v: str) -> str:
        if v not in cls.VALID_STATUSES:
            raise ValueError(f"Status must be one of {cls.VALID_STATUSES}")
        return v


class JobApplicationCreate(JobApplicationBase):
    """Schema for creating new job applications."""
    applied_date: Optional[datetime] = Field(
        None,
        description="Will be set to current time if not provided"
    )

    @field_validator('applied_date', mode='before')
    def parse_applied_date(cls: Self, value: Any) -> Optional[datetime]:
        if value == '' or value is None:
            return None
        return value


class JobApplicationDB(JobApplicationBase):
    """Complete job application schema including MongoDB ID."""
    id: ObjectId = Field(..., alias="_id", description="MongoDB document ID")

    model_config = ConfigDict(
        json_encoders={ObjectId: str},
        populate_by_name=True,
        arbitrary_types_allowed=True
    )


class JobApplication:
    """Handles CRUD operations for job applications in MongoDB."""
    
    @staticmethod
    def get_db() -> MongoDBConnection:
        return MongoDBConnection.get_db()
    
    @staticmethod
    def create(data: dict[str, str | datetime]) -> InsertOneResult:
        """Insert a new job application into the database."""
        app_data = JobApplicationCreate(**data).model_dump(exclude_unset=True)

        if not app_data.get('applied_date'):
            app_data['applied_date'] = datetime.now()
        return JobApplication.get_db().applications.insert_one(app_data)

    @staticmethod
    def get_all() -> list[JobApplicationDB]:
        """Retrieve all applications sorted by newest first."""
        return [
            JobApplicationDB(**app) 
            for app in JobApplication.get_db().applications.find().sort("applied_date", -1)
        ]

    @staticmethod
    def update(job_id: str, data: dict[str, str]) -> UpdateResult:
        """Update application fields."""
        update_data = JobApplicationCreate(**data).model_dump(exclude_unset=True)
        return JobApplication.get_db().applications.update_one(
            {"_id": ObjectId(job_id)},
            {"$set": update_data}
        )

    @staticmethod
    def delete(job_id: str) -> DeleteResult:
        """Delete an application by ID."""
        return JobApplication.get_db().applications.delete_one({"_id": ObjectId(job_id)})

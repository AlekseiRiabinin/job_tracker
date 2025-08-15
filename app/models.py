from flask import current_app
from datetime import datetime
from typing import Optional, Self, ClassVar, Any
from pydantic import (
    BaseModel, ConfigDict, Field,
    field_validator, model_validator
)
from pymongo.results import (
    InsertOneResult,
    UpdateResult,
    DeleteResult
)
from pymongo.database import Database
from bson import ObjectId


class JobApplicationBase(BaseModel):
    """Base schema for job application data validation."""
    VALID_STATUSES: ClassVar[set[str]] = {
        "Applied",
        "Interview/Phone",
        "Interview/Technical",
        "Interview/Onsite",
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
    source: str = Field(
        "Other", 
        description="Where (website) the job was found"
    )
    applied_date: datetime = Field(
        default_factory=datetime.now,
        description="When the application was submitted"
    )
    response_date: Optional[datetime] = Field(
        None, 
        description="When company responded"
    )
    response_days: Optional[int] = Field(
        None,
        description="Automatically calculated response time in days",
        exclude=True
    )
    vacancy_description: Optional[str] = Field(
        None,
        max_length=20000,
        description="Full text of the job vacancy/description",
        json_schema_extra={
            "mongo_index": "text",    # Enable text search in MongoDB
            "input_type": "textarea"  # Hint for frontend form
        }
    )
    ml_meta: dict = Field(
        default_factory=lambda: {
            "success_probability": None,
            "german_level": None,
            "last_prediction_date": None
        },
        exclude=True,
        description="ML system metadata",
        json_schema_extra={
            "frontend_visible": False,
            "mongo_field": "ml"   # Stores as subdocument
        }
    )

    @field_validator('status')
    def validate_status(cls: Self, v: str) -> str:
        if v not in cls.VALID_STATUSES:
            raise ValueError(f"Status must be one of {cls.VALID_STATUSES}")
        return v

    @model_validator(mode='after')
    def calculate_response_time(self: Self) -> Self:
        if self.applied_date and self.response_date:
            self.response_days = (self.response_date - self.applied_date).days
        return self


class JobApplicationCreate(JobApplicationBase):
    """Schema for creating new job applications."""
    applied_date: Optional[datetime] = Field(
        None,
        description="Will be set to current time if not provided"
    )
    response_date: Optional[datetime] = Field(
        None,
        description="Date when company responded (optional)"
    )

    @field_validator('applied_date', 'response_date', mode='before')
    def parse_dates(cls: Self, value: Any) -> Optional[datetime]:
        if value == '' or value is None:
            return None
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                raise ValueError("Invalid date format. Use YYYY-MM-DD")
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

    @classmethod
    def calculate_success_metrics(cls: Self, application: dict) -> dict:
        """Calculate success metrics based on application progress."""
        status = application['status']
        current_date = datetime.now()
        
        # Base success probability based on status hierarchy
        status_weights = {
            "Applied": 0.1,
            "Interview/Phone": 0.3,
            "Interview/Technical": 0.6,
            "Interview/Onsite": 0.8,
            "Offer": 1.0,
            "Rejected": 0.0,
            "Ghosted": 0.05
        }

        # Time decay factor (recent applications get more weight)
        days_since_applied = (current_date - application['applied_date']).days
        time_factor = max(0, 1 - (days_since_applied / 90))  # 3-month half-life

        # Response time bonus (faster responses = better odds)
        response_bonus = 0
        if application.get('response_date'):
            response_days = (
                application['response_date'] - application['applied_date']
            ).days
            response_bonus = 0.2 * (1 - min(response_days, 14) / 14)  # Up to 20% bonus

        # Calculate composite probability
        base_prob = status_weights.get(status, 0)
        composite_prob = min(1, base_prob * (1 + time_factor) / 2 + response_bonus)

        return {
            "success_probability": round(composite_prob, 2),
            "confidence_factor": time_factor,
            "last_calculated": current_date
        }

    @staticmethod
    def get_db() -> Database:
        """Get MongoDB instance from Flask app context."""
        if not current_app:
            raise RuntimeError("Not in Flask application context")
        return current_app.db

    @staticmethod
    def get_all() -> list[JobApplicationDB]:
        """Retrieve all applications sorted by newest first."""
        return [
            JobApplicationDB(**app) 
            for app in (
                JobApplication.get_db()
                    .applications
                    .find()
                    .sort("applied_date", -1)
            )
        ]

    @staticmethod
    def create(data: dict[str, str | datetime]) -> InsertOneResult:
        """Insert a new job application with auto-calculated ML metrics."""
        app_data = JobApplicationCreate(**data).model_dump(exclude_unset=True)

        if not app_data.get('applied_date'):
            app_data['applied_date'] = datetime.now()

        # Initialize ml_meta if not present
        app_data.setdefault('ml_meta', {})

        if 'status' in app_data:
            metrics = JobApplication.calculate_success_metrics({
                'status': app_data['status'],
                'applied_date': app_data['applied_date'],
                'response_date': app_data.get('response_date')
            })
            app_data['ml_meta'].update(metrics)

        return JobApplication.get_db().applications.insert_one(app_data)

    @staticmethod
    def update(job_id: str, data: dict[str, str]) -> UpdateResult:
        """Update application with recalculated ML metrics when status changes."""
        update_data = JobApplicationCreate(**data).model_dump(exclude_unset=True)
        
        if 'status' in update_data:
            current = JobApplication.get_db().applications.find_one(
                {"_id": ObjectId(job_id)},
                {"applied_date": 1, "response_date": 1, "ml_meta": 1}
            )

            if current:
                metrics = JobApplication.calculate_success_metrics({
                    'status': update_data['status'],
                    'applied_date': current.get('applied_date', datetime.now()),
                    'response_date': current.get('response_date')
                })

                # Preserve existing ml_meta fields not being updated
                update_data['ml_meta'] = {
                    **current.get('ml_meta', {}),
                    **metrics
                }

        return JobApplication.get_db().applications.update_one(
            {"_id": ObjectId(job_id)},
            {"$set": update_data}
        )

    @staticmethod
    def delete(job_id: str) -> DeleteResult:
        """Delete an application by ID."""
        return JobApplication.get_db().applications.delete_one(
            {"_id": ObjectId(job_id)}
        )

    @staticmethod
    def set_ml_prediction(
        job_id: str,
        probability: float,
        german_level: str
    ) -> None:
        """Safe update method for ML service."""
        JobApplication.get_db().applications.update_one(
            {"_id": ObjectId(job_id)},
            {"$set": {
                "ml.success_probability": probability,
                "ml.german_level": german_level,
                "ml.last_prediction_date": datetime.now()
            }}
        )

    @staticmethod
    def get_ml_training_data() -> list[dict]:
        """For job_predictor service only."""
        return [
            {
                "vacancy_description": app.vacancy_description,
                "role": app.role,
                "source": app.source,
                **app.ml_meta
            }
            for app in JobApplication.get_db().applications.find(
                {"vacancy_description": {"$exists": True}}
            )
        ]

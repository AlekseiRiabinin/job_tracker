import os
import time
from datetime import datetime
from pymongo import MongoClient
from pymongo.results import InsertOneResult, UpdateResult, DeleteResult
from pymongo.errors import ConnectionFailure, OperationFailure
from bson import ObjectId


def get_mongo_client() -> MongoClient:
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

try:
    client = get_mongo_client()
    db = client.jobtracker
except Exception as e:
    print(f"Failed to connect to MongoDB: {str(e)}")
    raise


class JobApplication:
    """Handles CRUD operations for job applications in MongoDB."""
    
    @staticmethod
    def create(data: dict[str, str | datetime]) -> InsertOneResult:
        """Insert a new job application into the database."""
        required_fields = {'company', 'location', 'role'}
        if missing := required_fields - set(data.keys()):
            raise KeyError(f"Missing required fields: {missing}")
            
        doc = {
            **data,
            'status': data.get('status', 'Applied'),
            'applied_date': datetime.now()
        }
        return db.applications.insert_one(doc)

    @staticmethod
    def get_all() -> list[dict[str, str | datetime]]:
        """Retrieve all applications sorted by newest first."""
        return list(db.applications.find().sort("applied_date", -1))

    @staticmethod
    def update(job_id: str, data: dict[str, str]) -> UpdateResult:
        """Update application fields."""
        return db.applications.update_one(
            {"_id": ObjectId(job_id)},
            {"$set": data}
        )

    @staticmethod
    def delete(job_id: str) -> DeleteResult:
        """Delete an application by ID."""
        return db.applications.delete_one({"_id": ObjectId(job_id)})

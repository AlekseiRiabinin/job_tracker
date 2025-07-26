from datetime import datetime
from pymongo import MongoClient
from bson import ObjectId
from pymongo.results import InsertOneResult, UpdateResult, DeleteResult


client = MongoClient(
    "mongodb://mongo:27017/",
    serverSelectionTimeoutMS=5000,
    connectTimeoutMS=30000,
    socketTimeoutMS=30000
)
db = client.jobtracker


class JobApplication:
    """Handles CRUD operations for job applications in MongoDB."""
    
    @staticmethod
    def create(data: dict[str, str | datetime]) -> InsertOneResult:
        """Insert a new job application into the database.
        
        Args:
            data: Must contain:
                - company: str
                - location: str 
                - role: str
                - status: str (default: 'Applied')
                - notes: Optional[str]
                
        Returns:
            InsertOneResult: MongoDB insertion result.
            
        Raises:
            KeyError: If required fields are missing.
        """
        required_fields = {'company', 'location', 'role'}
        if not required_fields.issubset(data.keys()):
            raise KeyError(f"Missing required fields: {required_fields - set(data.keys())}")
            
        data.setdefault('status', 'Applied')
        data['applied_date'] = datetime.now().strftime("%Y-%m-%d")
        return db.applications.insert_one(data)

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

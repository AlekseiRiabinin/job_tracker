import os
from datetime import datetime
from pymongo import MongoClient


client = MongoClient(os.getenv('MONGO_URI'))
db = client.jobtracker


def migrate():
    print(f"Starting migration at {datetime.now()}")
    
    result = db.applications.update_many(
        {"ml": {"$exists": False}},
        {"$set": {"ml": {
            "success_probability": None,
            "german_level": None,
            "last_prediction_date": None
        }}}
    )
    
    print(f"Successfully migrated {result.modified_count} documents")


if __name__ == "__main__":
    migrate()

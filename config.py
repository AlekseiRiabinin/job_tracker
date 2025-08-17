import os
from typing import Self
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    """App configuration with validation."""

    # MongoDB
    MONGO_USERNAME = os.getenv('MONGO_INITDB_ROOT_USERNAME')
    MONGO_PASSWORD = os.getenv('MONGO_INITDB_ROOT_PASSWORD')
    MONGO_HOST = os.getenv('MONGO_HOST', 'mongo')
    MONGO_PORT = int(os.getenv('MONGO_PORT', 27017))
    MONGO_DB_NAME = os.getenv('MONGO_DB_NAME', 'jobs_db')
    MONGO_MAX_RETRIES = int(os.getenv('MONGO_MAX_RETRIES', 5))
    MONGO_RETRY_DELAY = int(os.getenv('MONGO_RETRY_DELAY', 2))
    
    # ML
    ML_MODEL_DIR = str(Path(__file__).parent / 'services/job_predictor/models')
    TRAINING_COLLECTION = 'applications'
    MODEL_MAJOR_VERSION = int(os.getenv('MODEL_MAJOR_VERSION', 1))
    TRAIN_MODE = os.getenv('TRAIN_MODE', 'False').lower() == 'true'
    
    # Flask
    SECRET_KEY = os.getenv('FLASK_SECRET_KEY')
    API_VERSION = os.getenv('API_VERSION', '1.0')

    @property
    def MONGO_URI(self):
        return (
            f"mongodb://{self.MONGO_USERNAME}:{self.MONGO_PASSWORD}@"
            f"{self.MONGO_HOST}:{self.MONGO_PORT}/{self.MONGO_DB_NAME}?"
            "authSource=admin&retryWrites=true&w=majority"
        )

    @classmethod
    def validate(cls: Self) -> None:
        """Validate critical configurations."""
        if not cls.SECRET_KEY or cls.SECRET_KEY == 'admin':
            raise ValueError("FLASK_SECRET_KEY must be set and secure")
        
        required = [
            'MONGO_USERNAME',
            'MONGO_PASSWORD',
            'MONGO_HOST'
        ]
        for var in required:
            if not getattr(cls, var):
                raise ValueError(f"Missing required config: {var}")

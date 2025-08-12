import os
from dotenv import load_dotenv


load_dotenv()


class Config:
    MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/jobs_db')
    MONGO_DB_NAME = os.getenv('MONGO_DB_NAME', 'jobs_db')

    ML_MODEL_DIR = os.path.join(
        os.path.dirname(__file__), 
        'services/job_predictor/models'
    )
    TRAINING_COLLECTION = 'applications'

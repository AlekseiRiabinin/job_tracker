import time
from typing import Optional
from flask import Flask
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from flasgger import Swagger
from config import Config
from .services.job_predictor.predictor import JobPredictor


def create_app(test_config: Optional[dict] = None) -> Flask:
    """Factory function for creating the Flask application."""
    app = Flask(__name__)
    
    # Load configuration
    cfg = Config()
    app.config.from_object(cfg)
    if test_config:
        app.config.update(test_config)

    if 'MONGO_URI' not in app.config:
        raise RuntimeError("MONGO_URI not found in configuration")

    init_swagger(app)
    
    try:
        init_mongodb(app)
        init_predictor(app)
    except Exception as e:
        app.logger.error(f"Service initialization failed: {str(e)}")
        raise

    register_blueprints(app)
    
    return app


def init_swagger(app: Flask) -> None:
    """Initialize Swagger/OpenAPI documentation."""
    Swagger(app, template={
        'swagger': '2.0',
        'info': {
            'title': 'Job Tracker API',
            'description': 'API for job applications tracking',
            'version': app.config.get('API_VERSION', '1.0')
        },
        'consumes': ['application/json'],
        'produces': ['application/json'],
    })


def init_mongodb(app: Flask) -> None:
    """Initialize MongoDB connection with retry logic."""
    max_retries = app.config.get('MONGO_MAX_RETRIES', 5)
    retry_delay = app.config.get('MONGO_RETRY_DELAY', 2)
    
    for attempt in range(max_retries):
        try:
            app.logger.info(
                f"MongoDB connection attempt "
                f"{attempt + 1}/{max_retries}"
            )
            app.mongo_client = MongoClient(
                app.config['MONGO_URI'],
                serverSelectionTimeoutMS=5000,
                socketTimeoutMS=30000,
                connectTimeoutMS=30000,
                appname="job-tracker"
            )
            app.db = app.mongo_client[app.config['MONGO_DB_NAME']]
            app.db.command('ping')
            app.logger.info("MongoDB connection established")
            return
        except (ConnectionFailure, OperationFailure) as e:
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"MongoDB connection failed after "
                    f"{max_retries} attempts"
                ) from e
            time.sleep(retry_delay)


def init_predictor(app: Flask) -> None:
    """Initialize the ML predictor service."""
    app.predictor = JobPredictor(
        model_path=app.config['ML_MODEL_DIR'],
        major_version=app.config.get('MODEL_MAJOR_VERSION', 1),
        train_mode=app.config.get('TRAIN_MODE', False)
    )
    app.logger.info(
        f"Predictor initialized "
        f"(model v{app.predictor.model_version})"
    )


def register_blueprints(app: Flask) -> None:
    """Register all application blueprints."""
    from . import routes
    app.register_blueprint(routes.bp)

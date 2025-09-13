import time
from typing import Optional
from flask import Flask
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from flasgger import Swagger
from config import Config
from .services.job_predictor.predictor import JobPredictor


__all__ = ['create_app']


def create_app(test_config: Optional[dict] = None) -> Flask:
    """Factory function for creating the Flask application."""

    app = Flask(__name__)
    
    cfg = Config()
    app.config.from_object(cfg)
    if test_config:
        app.config.update(test_config)

    if 'MONGO_URI' not in app.config:
        raise RuntimeError(
            "MONGO_URI not found in configuration"
        )

    init_swagger(app)
    init_predictor(app)
    
    app.mongo_client = None
    app.db = None 
    
    register_cli_commands(app)
    
    @app.before_request
    def ensure_services_initialized():
        if (
            app.mongo_client is None and 
            not app.config.get('TESTING')
        ):
            try:
                init_mongodb(app)
                app.logger.info("Services initialized")

            except Exception as e:
                app.logger.error(
                    f"Service initialization failed: {str(e)}"
                )
                raise

    register_blueprints(app)
    return app


def register_cli_commands(app: Flask) -> None:
    """Register CLI commands as Flask commands."""

    from .cli import (
        collection_stats, load_from_json, load_from_csv, 
        export_to_json, export_to_csv
    )
    
    app.cli.add_command(collection_stats, name="collection-stats")
    app.cli.add_command(load_from_json, name="load-from-json")
    app.cli.add_command(load_from_csv, name="load-from-csv")
    app.cli.add_command(export_to_json, name="export-to-json")
    app.cli.add_command(export_to_csv, name="export-to-csv")


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

    if app.mongo_client is not None:
        return

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
                connect=False,
                appname="job-tracker"
            )
            
            # Force connection
            app.mongo_client.admin.command('ping')
            app.db = app.mongo_client[app.config['MONGO_DB_NAME']]
            
            app.logger.info("MongoDB connection established")
            return
            
        except (ConnectionFailure, OperationFailure) as e:
            if attempt == max_retries - 1:
                app.logger.error(
                    f"MongoDB connection failed after "
                    f"{max_retries} attempts"
                )
                raise RuntimeError(
                    f"MongoDB connection failed after "
                    f"{max_retries} attempts"
                ) from e
            time.sleep(retry_delay)
            continue

        except Exception as e:
            app.logger.error(
                f"Unexpected MongoDB error: {str(e)}"
            )
            raise


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


def get_db():
    """Access to database with lazy initialization."""

    from flask import current_app
    if current_app.db is None:
        init_mongodb(current_app)

    return current_app.db

import click
import time
from flask import Flask
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from flasgger import Swagger
from config import Config
from .services.data_loader import DataLoader
from .services.job_predictor.predictor import JobPredictor


def create_app() -> Flask:
    """Factory function for creating the Flask application."""
    app = Flask(__name__)
    app.config.from_object(Config)

    # Initialize extensions
    Swagger(app)

    # Initialize MongoDB with retry logic
    max_retries = 5
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            app.mongo_client = MongoClient(
                app.config['MONGO_URI'],
                serverSelectionTimeoutMS=5000,
                socketTimeoutMS=30000,
                connectTimeoutMS=30000
            )
            app.db = app.mongo_client[app.config['MONGO_DB_NAME']]
            app.db.command('ping')
            break
        except (ConnectionFailure, OperationFailure) as e:
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"Failed to connect to MongoDB after "
                    f"{max_retries} attempts: {str(e)}"
                )
            time.sleep(retry_delay)
    

    # Initialize ML predictor
    app.predictor = JobPredictor(
        model_path=app.config['ML_MODEL_DIR']
    )

    # Register blueprints
    from . import routes
    app.register_blueprint(routes.bp)
    
    app.cli.add_command(load_from_json)
    app.cli.add_command(load_from_csv)
    
    return app


@click.command()
@click.argument('file_path', type=click.Path(exists=True, dir_okay=False))
def load_from_json(file_path: str) -> None:
    """Load job applications from a JSON file."""
    try:
        DataLoader.load_from_json(file_path)
        click.echo(f"✅ Successfully loaded data from {file_path}")
    except Exception as e:
        click.echo(f"❌ Error loading data: {str(e)}", err=True)


@click.command()
@click.argument('file_path', type=click.Path(exists=True, dir_okay=False))
@click.option('--delimiter', default=',', help='CSV delimiter character')
@click.option('--encoding', default='utf-8', help='File encoding')
def load_from_csv(file_path: str, delimiter: str, encoding: str) -> None:
    """Load job applications from a CSV file."""
    try:
        DataLoader.load_from_csv(file_path, delimiter=delimiter, encoding=encoding)
        click.echo(f"✅ Successfully loaded data from {file_path}")
    except Exception as e:
        click.echo(f"❌ Error loading data: {str(e)}", err=True)


@click.group()
def cli() -> None:
    """Job Tracker CLI - Manage job applications data."""
    pass

cli.add_command(load_from_json)
cli.add_command(load_from_csv)


if __name__ == '__main__':
    cli()

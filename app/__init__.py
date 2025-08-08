import click
from flask import Flask
from flasgger import Swagger
from .services.data_loader import DataLoader


def create_app() -> Flask:
    """Factory function for creating the Flask application."""
    app = Flask(__name__)
    Swagger(app)
    
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

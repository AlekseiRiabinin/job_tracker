import click
from .services.data_loader import DataLoader


@click.group()
def cli():
    """Job Tracker CLI - Manage job applications data."""
    ...


@cli.command()
@click.argument('file_path', type=click.Path(exists=True, dir_okay=False))
def load_from_json(file_path: str) -> None:
    """Load job applications from a JSON file."""
    try:
        DataLoader.load_from_json(file_path)
        click.echo(f"✅ Successfully loaded data from {file_path}")
    except Exception as e:
        click.echo(f"❌ Error loading data: {str(e)}", err=True)


@cli.command()
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

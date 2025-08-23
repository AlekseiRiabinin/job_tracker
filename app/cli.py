import click
from pathlib import Path
from .services.data_loader import DataLoader
from .services.data_exporter import DataExporter


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


@cli.command()
@click.argument('file_path', type=click.Path(dir_okay=False))
@click.option('--collection', default='applications', help='MongoDB collection name')
@click.option('--query', default='{}', help='MongoDB query as JSON string')
def export_to_json(file_path: str, collection: str, query: str) -> None:
    """Export job applications to a JSON file."""
    try:
        import json as json_lib
        query_dict = json_lib.loads(query)
        
        count = DataExporter.export_to_json(file_path, collection, query_dict)
        click.echo(f"✅ Successfully exported {count} records to {file_path}")
    except Exception as e:
        click.echo(f"❌ Error exporting data: {str(e)}", err=True)


@cli.command()
@click.argument('file_path', type=click.Path(dir_okay=False))
@click.option('--collection', default='applications', help='MongoDB collection name')
@click.option('--query', default='{}', help='MongoDB query as JSON string')
@click.option('--delimiter', default=',', help='CSV delimiter character')
@click.option('--encoding', default='utf-8', help='File encoding')
def export_to_csv(
    file_path: str,
    collection: str,
    query: str,
    delimiter: str,
    encoding: str
) -> None:
    """Export job applications to a CSV file."""
    try:
        import json as json_lib
        query_dict = json_lib.loads(query)
        
        count = DataExporter.export_to_csv(
            file_path, collection, query_dict, delimiter, encoding
        )
        click.echo(f"✅ Successfully exported {count} records to {file_path}")
    except Exception as e:
        click.echo(f"❌ Error exporting data: {str(e)}", err=True)


@cli.command()
@click.argument('file_path', type=click.Path(dir_okay=False))
@click.option('--collection', default='applications', help='MongoDB collection name')
@click.option('--query', default='{}', help='MongoDB query as JSON string')
@click.option('--delimiter', default=',', help='CSV delimiter character')
@click.option('--encoding', default='utf-8', help='File encoding')
def export_to_csv(
    file_path: str,
    collection: str,
    query: str,
    delimiter: str,
    encoding: str
) -> None:
    """Export job applications to a CSV file."""
    try:
        import json as json_lib
        query_dict = json_lib.loads(query)
        
        count = DataExporter.export_to_csv(
            file_path, collection, query_dict, delimiter, encoding
        )
        click.echo(f"✅ Successfully exported {count} records to {file_path}")
    except Exception as e:
        click.echo(f"❌ Error exporting data: {str(e)}", err=True)

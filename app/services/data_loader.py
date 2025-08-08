import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Any
from ..models import JobApplication, JobApplicationCreate


class DataLoader:
    """Data loader."""

    @staticmethod
    def _convert_dates(item: dict[str, Any]) -> dict[str, Any]:
        """Helper method to convert string dates to datetime objects."""
        date_fields = ['applied_date', 'response_date']
        date_formats = ['%Y-%m-%d', '%Y-%m-%dT%H:%M:%S', '%m/%d/%Y']
        
        for field in date_fields:
            if field in item and isinstance(item[field], str):
                for fmt in date_formats:
                    try:
                        item[field] = datetime.strptime(item[field], fmt)
                        break
                    except ValueError:
                        continue
        return item

    @staticmethod
    def load_from_json(
        file_path: str,
        collection_name: str = "applications"
    ) -> None:
        """Load data from a JSON file."""  
        if not Path(file_path).exists():
            raise FileNotFoundError(f"JSON file not found at {file_path}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        db = JobApplication.get_db()
        collection = db[collection_name]

        for item in data:
            try:
                item = DataLoader._convert_dates(item)
                validated_data = JobApplicationCreate(**item).model_dump(exclude_unset=True)
                collection.insert_one(validated_data)
            except Exception as e:
                print(f"Error inserting data: {e}")

        print(f"Successfully loaded {len(data)} records from {file_path}")

    @staticmethod
    def load_from_csv(
        file_path: str,
        collection_name: str = "applications",
        delimiter: str = ",",
        encoding: str = "utf-8"
    ) -> None:
        """Load data from a CSV file."""
        if not Path(file_path).exists():
            raise FileNotFoundError(f"CSV file not found at {file_path}")
        
        db = JobApplication.get_db()
        collection = db[collection_name]
        loaded_count = 0

        with open(file_path, 'r', encoding=encoding) as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            
            for row in reader:
                try:
                    row = {k: v if v != '' else None for k, v in row.items()}
                    row = DataLoader._convert_dates(row)
                    
                    validated_data = JobApplicationCreate(**row).model_dump(exclude_unset=True)
                    collection.insert_one(validated_data)
                    loaded_count += 1
                except Exception as e:
                    print(f"Error inserting row {row}: {e}")

        print(f"Successfully loaded {loaded_count} records from {file_path}")

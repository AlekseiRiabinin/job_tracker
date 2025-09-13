import json
import csv
from datetime import datetime
from typing import Any, Optional
from bson import ObjectId
from .. import get_db


class DataExporter:
    """Data exporter for MongoDB collections."""

    @staticmethod
    def _serialize_dates(
        obj: datetime | ObjectId | Any
    ) -> str | Any:
        """Convert datetime objects to ISO format strings."""

        if isinstance(obj, datetime):
            return obj.isoformat()

        elif isinstance(obj, ObjectId):
            return str(obj)

        return obj

    @staticmethod
    def export_to_json(
        file_path: str,
        collection_name: str = "applications",
        query: Optional[dict[str, Any]] = None
    ) -> int:
        """Export data from MongoDB to a JSON file."""

        db = get_db()
        collection = db[collection_name]
        
        query = query or {}
        data = list(collection.find(query))
        
        serialized_data = []
        for item in data:
            item_dict = dict(item)
            serialized_item = {
                key: DataExporter._serialize_dates(value) 
                for key, value in item_dict.items()
            }
            serialized_data.append(serialized_item)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(
                serialized_data,
                f,
                indent=2,
                default=str,
                ensure_ascii=False
            )
        
        return len(serialized_data)

    @staticmethod
    def export_to_csv(
        file_path: str,
        collection_name: str = "applications",
        query: Optional[dict] = None,
        delimiter: str = ",",
        encoding: str = "utf-8"
    ) -> int:
        """Export data from MongoDB to a CSV file."""

        db = get_db()
        collection = db[collection_name]
        
        query = query or {}
        data = list(collection.find(query))
        
        if not data:
            return 0
        
        data_as_dicts = [dict(item) for item in data]
        
        fieldnames = set()
        for item in data_as_dicts:
            fieldnames.update(item.keys())
        fieldnames = sorted(fieldnames)
        
        with open(
            file_path, 'w', encoding=encoding, newline=''
        ) as f:
            writer = csv.DictWriter(
                f, fieldnames=fieldnames, delimiter=delimiter
            )
            writer.writeheader()
            
            for item in data_as_dicts:
                serialized_item = {}
                for key, value in item.items():
                    serialized_item[key] = (
                        DataExporter._serialize_dates(value)
                    )
                writer.writerow(serialized_item)
        
        return len(data)

    @staticmethod
    def get_collection_stats(
        collection_name: str = "applications"
    ) -> dict:
        """Get statistics about a collection."""

        db = get_db()
        collection = db[collection_name]
        
        total_count = collection.count_documents({})
        sample_doc = collection.find_one()

        return {
            "collection_name": collection_name,
            "total_records": total_count,
            "fields": (
                list(dict(sample_doc).keys()) 
                if total_count > 0 else []
            )
        }

from datetime import datetime, timedelta
from collections import defaultdict
from ..models import JobApplication


class AnalyticsService:
    """Visualization of Job Tracker data."""
    @staticmethod
    def get_summary_stats() -> dict:
        """Generate cards for dashboard header."""
        applications = JobApplication.get_db().applications
        total = applications.count_documents({})
        
        pipeline = [
            {"$group": {
                "_id": "$status",
                "count": {"$sum": 1}
            }}
        ]
        status_counts = dict(
            (res["_id"], res["count"])
            for res in applications.aggregate(pipeline)
        )

        return {
            "total_applications": total,
            "interviews": status_counts.get("Interview", 0),
            "offers": status_counts.get("Offer", 0),
            "rejection_rate": (
                status_counts.get("Rejected", 0) / total * 100 
                if total > 0 else 0
            )
        }

    @staticmethod
    def get_status_distribution() -> dict:
        """For pie chart - status percentages"""
        # Uses same pipeline as above
        ...

    @staticmethod
    def get_timeseries(
        start_date: str,
        end_date: str | None = None,
        include_applications: bool = False
    ) -> list[dict]:
        """Get daily application counts between dates (inclusive)"""
        date_filter = {"applied_date": {"$gte": datetime.fromisoformat(start_date)}}
        if end_date:
            date_filter["applied_date"]["$lte"] = datetime.fromisoformat(end_date)

        group_stage = {
            "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$applied_date"}},
            "count": {"$sum": 1}
        }
        
        if include_applications:
            group_stage["applications"] = {
                "$push": {
                    "id": {"$toString": "$_id"},
                    "company": "$company",
                    "role": "$role",
                    "status": "$status"
                }
            }

        pipeline = [
            {"$match": date_filter},
            {"$group": group_stage},
            {"$project": {
                "date": "$_id",
                "count": 1,
                **({"applications": 1} if include_applications else {}),
                "_id": 0
            }},
            {"$sort": {"date": 1}}
        ]

        return list(JobApplication.get_db().applications.aggregate(pipeline))

    @staticmethod
    def get_response_times() -> dict:
        """For bar chart - company response times"""
        # Requires adding 'response_date' field to schema
        ...

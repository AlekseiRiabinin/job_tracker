from datetime import datetime, timedelta
from ..models import JobApplication


class AnalyticsService:
    """Advanced analytics for job applications."""

    @staticmethod
    def get_summary_stats() -> dict:
        """Generate cards for dashboard header."""
        applications = JobApplication.get_db().applications
        total = applications.count_documents({})
        
        pipeline = [
            {"$group": {
                "_id": "$status",
                "count": {"$sum": 1},
                "avg_response": {"$avg": "$response_days"}
            }}
        ]

        results = list(applications.aggregate(pipeline))
        status_counts = {r["_id"]: r["count"] for r in results}

        total_interviews = sum(
            count for status, count in status_counts.items() 
            if status.startswith("Interview/")
        )

        return {
            "total_applications": total,
            "by_status": status_counts,
            "conversion_rates": {
                "app_to_interview": (
                    total_interviews / total * 100 
                    if total > 0 else 0
                ),
                "interview_to_offer": (
                    status_counts.get("Offer", 0) / total_interviews * 100
                    if total_interviews > 0 else 0
                ),
                "app_to_offer": (
                    status_counts.get("Offer", 0) / total * 100
                    if total > 0 else 0
                )
            },
            "avg_response_times": {
                r["_id"]: r["avg_response"] 
                for r in results if r["avg_response"]
            }
        }

    @staticmethod
    def get_status_distribution() -> dict:
        """Detailed status distribution with percentages."""
        stats = AnalyticsService.get_summary_stats()
        total = stats["total_applications"]

        status_details = {
            status: {
                "count": count,
                "percentage": count / total * 100 if total > 0 else 0,
                "type": "offer" if status == "Offer" else 
                    "rejection" if status in ("Rejected", "Ghosted") else
                    "interview" if status.startswith("Interview/") else
                    "application"
            }
            for status, count in stats["by_status"].items()
        }
        
        interview_stages = {
            "total_interviews": sum(
                count for status, count in stats["by_status"].items() 
                if status.startswith("Interview/")
            ),
            "breakdown": {
                status: status_details[status]
                for status in stats["by_status"]
                if status.startswith("Interview/")
            }
        }
        
        return {
            "individual_statuses": status_details,
            "interview_pipeline": interview_stages,
            "summary": {
                "applied": stats["by_status"].get("Applied", 0),
                "in_interview": interview_stages["total_interviews"],
                "offers": stats["by_status"].get("Offer", 0),
                "rejections": (
                    stats["by_status"].get("Rejected", 0) + 
                    stats["by_status"].get("Ghosted", 0)
                )
            },
            "conversion_rates": stats["conversion_rates"]
        }

    @staticmethod
    def get_response_metrics() -> dict:
        """Comprehensive response time analysis."""
        pipeline = [
            {"$match": {"response_days": {"$exists": True}}},
            {"$group": {
                "_id": None,
                "avg_response": {"$avg": "$response_days"},
                "median_response": {
                    "$median": {
                        "input": "$response_days", 
                        "method": "approximate"
                    }
                },
                "companies": {
                    "$push": {
                        "name": "$company",
                        "response_days": "$response_days"
                    }
                }
            }},
            {"$project": {
                "_id": 0,
                "avg_response": 1,
                "median_response": 1,
                "fastest_responders": {
                    "$slice": [
                        {"$sortArray": {
                            "input": "$companies", 
                            "sortBy": {"response_days": 1}
                        }},
                        5
                    ]
                },
                "slowest_responders": {
                    "$slice": [
                        {"$sortArray": {
                            "input": "$companies", 
                            "sortBy": {"response_days": -1}
                        }},
                        5
                    ]
                }
            }}
        ]
        
        result = list(JobApplication.get_db().applications.aggregate(pipeline))
        return result[0] if result else {}

    @staticmethod
    def get_timeseries(
        start_date: str,
        end_date: str | None = None,
        include_applications: bool = False,
        status_filter: str | None = None
    ) -> list[dict]:
        """Get daily application counts between dates (inclusive)."""
        date_filter = {"applied_date": {"$gte": datetime.fromisoformat(start_date)}}
        if end_date:
            date_filter["applied_date"]["$lte"] = datetime.fromisoformat(end_date)
        if status_filter:
            date_filter["status"] = status_filter

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

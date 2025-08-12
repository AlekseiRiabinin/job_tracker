from datetime import datetime, timezone
from flask import (
    Blueprint, current_app, request,
    redirect, url_for, abort, jsonify,
    render_template
)
from bson import ObjectId
from bson.errors import InvalidId
from .models import JobApplication, JobApplicationDB
from .services.analytics import AnalyticsService
from .services.job_predictor.predictor import JobPredictor


bp = Blueprint('main', __name__)

predictor = JobPredictor()


def is_api_request() -> bool:
    """Check if request prefers JSON (API) or HTML (web)."""
    return (
        request.args.get('format') == 'json' or
        (request.accept_mimetypes.accept_json and 
         not request.accept_mimetypes.accept_html)
    )


@bp.route('/', methods=['GET'])
def index() -> str | dict:
    """Display all applications (HTML or JSON)."""
    try:
        jobs = JobApplication.get_all()
        
        if is_api_request():
            serialized_jobs = []
            for job in jobs:
                job_data = job.model_dump()
                job_data['id'] = str(job_data['id'])
                serialized_jobs.append(job_data)
            return jsonify(serialized_jobs)
            
        return render_template('index.html', jobs=jobs)
        
    except Exception as e:
        if is_api_request():
            return jsonify({
                "error": "Failed to fetch jobs",
                "details": str(e)
            }), 500
        abort(500)


@bp.route('/add', methods=['GET', 'POST'])
def add_job() -> str | dict:
    """Handle job application creation."""
    if request.method == 'POST':
        try:
            data = (
                request.get_json() 
                if request.is_json 
                else request.form.to_dict()
            )

            if 'response_date' in data and data['response_date'] == '':
                data['response_date'] = None
            if 'applied_date' in data and data['applied_date'] == '':
                data['applied_date'] = None

            result = JobApplication.create(data)
            
            if is_api_request():
                return jsonify({
                    "status": "success",
                    "id": str(result.inserted_id)
                }), 201
                
            return redirect(url_for('main.index'))
            
        except ValueError as e:
            if is_api_request():
                return jsonify({
                    "error": "Validation failed",
                    "details": str(e)
                }), 400
            abort(400, description=str(e))
    
    if is_api_request():
        return jsonify({"error": "Use POST to create jobs"}), 405
    return render_template('add_job.html')


@bp.route('/edit/<job_id>', methods=['GET', 'POST'])
def edit_job(job_id: str) -> str | dict:
    """Handle editing job applications."""
    try:
        job = JobApplication.get_db().applications.find_one({"_id": ObjectId(job_id)})
        if not job:
            if is_api_request():
                return jsonify({"error": "Job not found"}), 404
            abort(404, "Job application not found")

        if request.method == 'POST':
            try:
                data = (
                    request.get_json() 
                    if request.is_json 
                    else request.form.to_dict()
                )
                JobApplication.update(job_id, data)
                
                if is_api_request():
                    return jsonify({"status": "success"})
                return redirect(url_for('main.index'))
                
            except ValueError as e:
                if is_api_request():
                    return jsonify({
                        "error": "Validation failed",
                        "details": str(e)
                    }), 400
                abort(400, description=str(e))

        job = JobApplicationDB(**job)
        if is_api_request():
            job_data = job.model_dump()
            job_data['id'] = str(job_data['id'])
            return jsonify(job_data)
            
        return render_template('edit_job.html', job=job.model_dump())

    except InvalidId:
        if is_api_request():
            return jsonify({"error": "Invalid ID format"}), 400
        abort(404, "Invalid job ID format")


@bp.route('/delete/<job_id>', methods=['DELETE', 'GET'])
def delete_job(job_id: str) -> str | dict:
    """Handle application deletion."""
    try:
        JobApplication.delete(job_id)
        
        if request.method == 'DELETE' or is_api_request():
            return jsonify({"status": "success"})
            
        return redirect(url_for('main.index'))
        
    except InvalidId:
        if is_api_request():
            return jsonify({"error": "Invalid ID format"}), 400
        abort(404, "Invalid job ID")


@bp.route('/analytics/summary', methods=['GET'])
def analytics_summary() -> str | dict:
    """Display comprehensive summary statistics for job applications."""
    try:
        stats = {
            "basic_stats": AnalyticsService.get_summary_stats(),
            "status_distribution": AnalyticsService.get_status_distribution(),
            "response_metrics": AnalyticsService.get_response_metrics()
        }

        if is_api_request():
            return jsonify({
                "meta": {"generated_at": datetime.now(timezone.utc).isoformat()},
                "data": stats
            })

        return render_template(
            'dashboard/overview.html',
            stats=stats,
            title="Application Analytics Summary"
        )

    except Exception as e:
        error_msg = "Failed to generate analytics summary"
        current_app.logger.error(f"{error_msg}: {str(e)}")
        if is_api_request():
            return jsonify({
                "error": error_msg,
                "details": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }), 500
        abort(500)


@bp.route('/analytics/timeseries', methods=['GET'])
def analytics_timeseries() -> str | dict:
    """Display time-series data of applications."""
    try:
        start_date = request.args.get(
            'start',
            default=f"{datetime.now().year}-01-01"
        )
        end_date = request.args.get('end')
        include_details = (
            request.args.get('details', '').lower() 
            in ('true', '1', 'yes')
        )

        data = AnalyticsService.get_timeseries(
            start_date=start_date,
            end_date=end_date,
            include_applications=include_details
        )
        
        if is_api_request():
            return jsonify({
                "meta": {
                    "start_date": start_date,
                    "end_date": end_date,
                    "count": len(data)
                },
                "data": data
            })

        return render_template(
            'dashboard/timeseries.html',
            chart_data=data,
            chart_labels=[entry['date'] for entry in data],
            chart_values=[entry['count'] for entry in data],
            show_details=include_details,
            title=f"Applications from {start_date}" + 
                 (f" to {end_date}" if end_date else "")
        )
        
    except ValueError as e:
        error_msg = f"Invalid date format: {str(e)}. Use YYYY-MM-DD."
        if is_api_request():
            return jsonify({"error": error_msg}), 400
        abort(400, description=error_msg)
        
    except Exception as e:
        error_msg = "Failed to fetch timeseries data"
        current_app.logger.error(f"{error_msg}: {str(e)}")
        if is_api_request():
            return jsonify({"error": error_msg}), 500
        abort(500)


@bp.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        german_level = data.get('german_level')
        proba = predictor.predict(data, german_level)
        return jsonify({'probability': proba, 'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 400


@bp.route('/retrain', methods=['POST'])
def retrain():
    try:
        metrics = current_app.predictor.train_from_mongodb()
        return jsonify({
            'status': 'success',
            'metrics': metrics
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

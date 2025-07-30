from flask import (
    Blueprint, render_template, request,
    redirect, url_for, abort, jsonify
)
from bson import ObjectId
from bson.errors import InvalidId
from .models import JobApplication, JobApplicationDB


bp = Blueprint('main', __name__)


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

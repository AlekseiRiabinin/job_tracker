from flask import Blueprint, render_template, request, redirect, url_for, abort
from werkzeug.wrappers.response import Response
from bson import ObjectId
from bson.errors import InvalidId
from .models import JobApplication, JobApplicationDB


bp = Blueprint('main', __name__)


@bp.route('/')
def index() -> str:
    """Display all applications sorted by date."""
    jobs = JobApplication.get_all()
    return render_template('index.html', jobs=jobs)


@bp.route('/add', methods=['GET', 'POST'])
def add_job() -> str | Response:
    """Handle job application creation."""
    if request.method == 'POST':
        try:
            JobApplication.create(request.form.to_dict())
            return redirect(url_for('main.index'))
        except ValueError as e:
            abort(400, description=str(e))
    return render_template('add_job.html')


@bp.route('/edit/<job_id>', methods=['GET', 'POST'])
def edit_job(job_id: str) -> str | Response:
    """Handle editing existing job applications."""
    try:
        job = JobApplication.get_db().applications.find_one({"_id": ObjectId(job_id)})
        if not job:
            abort(404, "Job application not found")

        if request.method == 'POST':
            try:
                JobApplication.update(job_id, request.form.to_dict())
                return redirect(url_for('main.index'))
            except ValueError as e:
                abort(400, description=str(e))

        job = JobApplicationDB(**job)
        return render_template('edit_job.html', job=job.model_dump())

    except InvalidId:
        abort(404, "Invalid job ID format")


@bp.route('/delete/<job_id>')
def delete_job(job_id: str) -> Response:
    """Handle application deletion."""
    try:
        JobApplication.delete(job_id)
    except InvalidId:
        abort(404, "Invalid job ID")
    return redirect(url_for('main.index'))

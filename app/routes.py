from flask import Flask, render_template, request, redirect, url_for, abort
from werkzeug.wrappers.response import Response
from bson import ObjectId
from bson.errors import InvalidId
from .models import JobApplication, db


app = Flask(__name__)


@app.route('/')
def index() -> str:
    """Display all applications sorted by date."""
    jobs = JobApplication.get_all()
    return render_template('index.html', jobs=jobs)


@app.route('/add', methods=['GET', 'POST'])
def add_job() -> str | Response:
    """Handle job application creation."""
    if request.method == 'POST':
        try:
            JobApplication.create({
                "company": request.form['company'],
                "location": request.form['location'],
                "role": request.form['role'],
                "status": request.form.get('status', 'Applied'),
                "notes": request.form.get('notes', '')
            })
            return redirect(url_for('index'))
        except KeyError as e:
            abort(400, description=f"Missing required field: {e.args[0]}")
    return render_template('add_job.html')


@app.route('/edit/<job_id>', methods=['GET', 'POST'])
def edit_job(job_id: str) -> str | Response:
    """Handle editing existing job applications."""
    try:
        job = db.applications.find_one({"_id": ObjectId(job_id)})
        if not job:
            abort(404, "Job application not found")

        if request.method == 'POST':
            try:
                JobApplication.update(job_id, {
                    "status": request.form['status'],
                    "notes": request.form['notes']
                })
                return redirect(url_for('index'))
            except KeyError as e:
                abort(400, description=f"Missing required field: {e.args[0]}")

        job['_id'] = str(job['_id'])
        return render_template('edit_job.html', job=job, job_id=str(job['_id']))

    except InvalidId:
        abort(404, "Invalid job ID format")


@app.route('/delete/<job_id>')
def delete_job(job_id: str) -> Response:
    """Handle application deletion."""
    try:
        JobApplication.delete(job_id)
    except InvalidId:
        abort(404, "Invalid job ID")
    return redirect(url_for('index'))

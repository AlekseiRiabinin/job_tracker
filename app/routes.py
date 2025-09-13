import traceback
from flask import (
    Blueprint, Response, current_app, request,
    redirect, url_for, abort, jsonify,
    render_template, flash
)
from datetime import datetime, timezone
from bson import ObjectId
from bson.errors import InvalidId
from werkzeug.wrappers.response import Response as WerkzeugResponse
from .models import JobApplication, JobApplicationDB
from .services.analytics import AnalyticsService


bp = Blueprint('main', __name__)

type FlaskResponse = Response | WerkzeugResponse | str


def is_api_request() -> bool:
    """Check if request prefers JSON (API) or HTML (web)."""

    return (
        request.args.get('format') == 'json' or
        request.headers.get('Content-Type') == 'application/json' or
        (request.accept_mimetypes.accept_json and 
         not request.accept_mimetypes.accept_html)
    )


@bp.route('/', methods=['GET'])
def index() -> FlaskResponse:
    """Display all applications with proper error handling."""

    try:
        if (
            not hasattr(current_app, 'db') or 
            current_app.db is None
        ):
            raise RuntimeError("Database not initialized")
        
        jobs = JobApplication.get_all()
        
        if is_api_request():
            return jsonify([
                {**job.model_dump(), "id": str(job.id)} 
                for job in jobs
            ])
            
        return render_template('index.html', jobs=jobs)
        
    except Exception as e:
        current_app.logger.error(
            f"Index route failed: "
            f"{str(e)}\n{traceback.format_exc()}"
        )
        if is_api_request():
            return jsonify({
                "status": "error",
                "error": "Failed to load applications",
                "details": str(e)
            }), 500
        flash("Failed to load applications", "danger")
        abort(500)


@bp.route('/add', methods=['GET', 'POST'])
def add_job() -> FlaskResponse:
    """Handle job application creation."""

    if request.method == 'POST':
        try:
            data = (
                request.get_json() 
                if request.is_json 
                else request.form.to_dict()
            )

            if (
                'response_date' in data and 
                data['response_date'] == ''
            ):
                data['response_date'] = None

            if (
                'applied_date' in data and 
                data['applied_date'] == ''
            ):
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
def edit_job(job_id: str) -> FlaskResponse:
    """Handle editing job applications."""

    try:
        job = JobApplication.get_db().applications.find_one(
            {"_id": ObjectId(job_id)}
        )
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
def delete_job(job_id: str) -> FlaskResponse:
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
def analytics_summary() -> FlaskResponse:
    """Display summary statistics for job applications."""

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


@bp.route('/ml/predict/one', methods=['GET', 'POST'])
def ml_predict() -> FlaskResponse:
    """Handle prediction requests."""

    # Service availability check
    if (
        not hasattr(current_app, 'predictor') 
        or current_app.predictor is None
    ):
        error_msg = "Prediction service is not available"
        if is_api_request():
            return jsonify(
                {'status': 'error', 'error': error_msg}
            ), 503

        flash(error_msg, 'danger')
        return redirect(url_for('main.index'))

    # Model readiness check
    model_ready = (
        hasattr(current_app.predictor, 'is_ready') and 
        current_app.predictor.is_ready()
    )
    model_version = getattr(
        current_app.predictor, 'model_version', '0.0'
    )

    # GET Request Handling
    if request.method == 'GET':
        base_vars = {
            'model_ready': model_ready,
            'model_version': model_version,
            'prediction': None,
            'error': None
        }

        if is_api_request():
            return jsonify({
                'status': 'success' if model_ready else 'error',
                'model_ready': model_ready,
                'model_version': model_version,
                'message': (
                    'Model is ready' 
                    if model_ready 
                    else 'Model not trained yet'
                )
            }), (200 if model_ready else 503)
            
        return render_template('dashboard/predict.html', **base_vars)

    # POST Request Handling
    if not model_ready:
        error_msg = (
            f"Model not trained yet. "
            f"Please retrain the model first."
        )
        if is_api_request():
            return jsonify({
                'status': 'error',
                'error': error_msg,
                'solution': 'Retrain the model first',
                'retrain_endpoint': url_for('main.ml_retrain')
            }), 503

        flash(error_msg, 'warning')
        return redirect(url_for('main.ml_retrain'))

    # Process Prediction Request
    form_data = (
        request.get_json() 
        if is_api_request() 
        else request.form
    )
    
    data = {
        'vacancy_description': form_data.get('description'),
        'role': form_data.get('role'),
        'source': form_data.get('source'),
        'german_level': form_data.get('german_level')
    }

    # Validate Required Fields
    required_fields = {'vacancy_description', 'role', 'source'}
    if not all(field in data for field in required_fields):
        error_msg = (
            f"Missing required fields: "
            f"{required_fields - set(data.keys())}"
        )
        if is_api_request():
            return jsonify({
                'status': 'error',
                'error': error_msg,
                'missing_fields': (
                    list(required_fields - set(data.keys()))
                )
            }), 400

        flash(error_msg, 'warning')
        return redirect(url_for('main.ml_predict'))

    # Attempt Prediction
    try:
        proba = current_app.predictor.predict(
            job_data={
                'vacancy_description': data['vacancy_description'],
                'role': data['role'],
                'source': data['source']
            },
            german_level=data.get('german_level')
        )

        if is_api_request():
            return jsonify({
                'status': 'success',
                'probability': float(proba),
                'model_version': model_version,
                'prediction_meta': {
                    'threshold': 0.5,
                    'result': 'Probability of application success'
                }
            })

        return render_template(
            'dashboard/predict.html',
            prediction={
                'probability': float(proba),
                'model_version': model_version
            },
            model_ready=True,
            error=None
        )

    except ValueError as e:
        error_msg = f"Input validation error: {str(e)}"
        if is_api_request():
            return jsonify({
                'status': 'error',
                'error': error_msg,
                'type': 'validation_error'
            }), 400

        flash(error_msg, 'warning')
        return redirect(url_for('main.ml_predict'))
        
    except Exception as e:
        current_app.logger.error(
            f"Prediction failed: "
            f"{str(e)}\n{traceback.format_exc()}"
        )
        error_msg = "Prediction service encountered an error"
        if is_api_request():
            return jsonify({
                'status': 'error',
                'error': error_msg,
                'error_details': str(e),
                'support_contact': 'aleksei.riabinin@yahoo.com'
            }), 500

        flash(error_msg, 'danger')
        return redirect(url_for('main.ml_predict'))


@bp.route('/ml/predict/batch', methods=['POST'])
def ml_batch_predict() -> FlaskResponse:
    """Handle batch prediction requests."""
    
    # Service availability check
    if (
        not hasattr(current_app, 'predictor') 
        or current_app.predictor is None
    ):
        error_msg = "Prediction service is not available"
        if is_api_request():
            return jsonify(
                {'status': 'error', 'error': error_msg}
            ), 503

        flash(error_msg, 'danger')
        return redirect(url_for('main.index'))

    # Model readiness check
    model_ready = (
        hasattr(current_app.predictor, 'pipeline') and 
        current_app.predictor.pipeline is not None
    )
    model_version = getattr(
        current_app.predictor, 'model_version', '0.0'
    )

    if not model_ready:
        error_msg = (
            f"Model not trained yet. "
            f"Please retrain the model first."
        )
        if is_api_request():
            return jsonify({
                'status': 'error',
                'error': error_msg,
                'solution': 'Retrain the model first',
                'retrain_endpoint': url_for('main.ml_retrain')
            }), 503

        flash(error_msg, 'warning')
        return redirect(url_for('main.ml_retrain'))

    # Process Request
    try:
        request_data = (
            request.get_json() 
            if is_api_request() 
            else request.form
        )
        
        # Validate input format
        if not request_data:
            error_msg = "No input data provided"
            if is_api_request():
                return jsonify({
                    'status': 'error',
                    'error': error_msg,
                    'required_format': {
                        'filter': {
                            'optional': 'MongoDB query filter'
                        },
                        'batch_size': {
                            'optional': 'Number of documents per batch'
                        },
                        'update_threshold': {
                            'optional': 'Min probability change for update'
                        }
                    }
                }), 400

            flash(error_msg, 'warning')
            return redirect(url_for('main.ml_batch_predict'))

        # Process batch prediction
        results = current_app.predictor.predict_batch(
            query_filter=request_data.get('filter'),
            batch_size=int(request_data.get('batch_size', 100)),
            update_threshold=float(request_data.get('update_threshold', 0.01)),
            max_documents=int(request_data.get('max_documents', 0)) or None
        )

        # Prepare response
        response_data = {
            'status': 'success',
            'model_version': model_version,
            'stats': {
                'processed': results['processed'],
                'updated': results['updated'],
                'skipped': results['skipped'],
                'errors': results['errors']
            },
            'timestamp': datetime.now().isoformat()
        }

        if results['errors'] > 0:
            response_data['warning'] = (
                f"{results['errors']} predictions failed"
            )
            if is_api_request():
                response_data['error_samples'] = (
                    results['error_details'][:5]
                )

        if is_api_request():
            return jsonify(response_data)

        flash(
            f"Batch prediction complete: "
            f"{results['processed']} processed, "
            f"{results['updated']} updated", 
            'success'
        )
        return render_template(
            'dashboard/batch_predict.html',
            results=response_data,
            model_ready=True
        )

    except ValueError as e:
        error_msg = f"Input validation error: {str(e)}"
        if is_api_request():
            return jsonify({
                'status': 'error',
                'error': error_msg,
                'type': 'validation_error'
            }), 400

        flash(error_msg, 'warning')
        return redirect(url_for('main.ml_batch_predict'))
        
    except Exception as e:
        current_app.logger.error(
            f"Batch prediction failed: "
            f"{str(e)}\n{traceback.format_exc()}"
        )
        error_msg = "Batch prediction service encountered an error"
        if is_api_request():
            return jsonify({
                'status': 'error',
                'error': error_msg,
                'error_details': str(e),
                'support_contact': 'aleksei.riabinin@yahoo.com'
            }), 500

        flash(error_msg, 'danger')
        return redirect(url_for('main.ml_batch_predict'))


@bp.route('/ml/retrain', methods=['GET', 'POST'])
def ml_retrain() -> FlaskResponse:
    """Handle retraining requests from both web and API."""

    if (
        not hasattr(current_app, 'predictor') 
        or current_app.predictor is None
    ):
        error_msg = "ML service is not available"
        if is_api_request():
            return jsonify({'status': 'error', 'error': error_msg}), 503

        flash(error_msg, 'danger')
        return redirect(url_for('main.index'))

    if request.method == 'POST':
        if (
            not is_api_request() and 
            not request.form.get('confirm')
        ):
            flash("You must confirm the retraining operation", 'warning')
            return redirect(url_for('main.ml_retrain'))

        try:
            metrics = current_app.predictor.train_from_mongodb()

            if is_api_request():
                return jsonify({
                    'status': 'success',
                    'metrics': metrics,
                    'model_version': metrics.get('model_version')
                })

            return render_template(
                'dashboard/retrain.html',
                current_model={
                    'version': metrics['model_version'],
                    'last_retrained': datetime.now().isoformat(),
                    'model_type': 'GradientBoostingClassifier'
                },
                training_result=metrics)
        
        except ValueError as e:
            error_msg = f"Validation error: {str(e)}"
            if is_api_request():
                return jsonify({
                    'status': 'error',
                    'error': error_msg,
                    'type': 'validation_error'
                }), 400

            flash(error_msg, 'warning')
            return redirect(url_for('main.ml_retrain'))
            
        except Exception as e:
            current_app.logger.error(
                f"Retraining failed: "
                f"{str(e)}\n{traceback.format_exc()}"
            )
            error_msg = "Model retraining failed"
            if is_api_request():
                return jsonify({
                    'status': 'error',
                    'error': error_msg,
                    'details': str(e)
                }), 500

            flash(error_msg, 'danger')
            return redirect(url_for('main.ml_retrain'))

    # GET request - show current model status
    model_ready = (
        hasattr(current_app.predictor, 'pipeline') and 
        current_app.predictor.pipeline is not None
    )
    
    model_info = {
        'version': current_app.predictor.model_version,
        'last_retrained': (
            datetime.now().isoformat() 
            if model_ready else "Never"
        ),
        'model_type': 'GradientBoostingClassifier',
        'is_ready': model_ready
    }

    if is_api_request():
        return jsonify({
            'status': 'success',
            'model_info': model_info,
            'message': (
                'Model is ready' if model_ready 
                else 'Model not trained yet'
            )
        })

    return render_template(
        'dashboard/retrain.html', 
        current_model=model_info
    )

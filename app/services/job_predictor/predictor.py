import os
import re
import sys
import joblib
import traceback
import logging
import pandas as pd
from flask import current_app
from datetime import datetime
from typing import Self, Optional
from sklearn.pipeline import Pipeline ,make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from .feature_engine import (
    TechStackTransformer,
    LanguageAwareTfidf,
    create_enhanced_features
)


class JobPredictor:
    """ML class for data loading, training and making predictions."""

    def __init__(
        self: Self,
        model_path: str = None, 
        train_mode: bool = False, 
        db_config: Optional[dict] = None,
        major_version: int = 1
    ) -> None:
        """Initialize prediction service with model path checking."""

        self.logger = logging.getLogger(__name__)
        self.major_version = major_version
        self.model_version = f"{major_version}.0"
        self.feature_processor = create_enhanced_features

        self._model_dir = (
            os.path.abspath(model_path) 
            if model_path
            else os.path.join(os.path.dirname(__file__), 'models')
        )
        os.makedirs(self._model_dir, exist_ok=True)

        self.pipeline = None
        available_models = self._get_available_models()
        self.logger.info(f"Available models: {available_models}")

        if len(available_models) > 0 and not train_mode:
            latest_model_path = self._get_latest_model()
            self.logger.info(f"Latest model path: {latest_model_path}")

            if latest_model_path:             
                try:
                    self.pipeline = self._load_model(latest_model_path)
                    if hasattr(self.pipeline, 'predict'):
                        self.logger.info(
                            f"Successfully loaded model from "
                            f"{os.path.basename(latest_model_path)} "
                            f"(version {self.model_version})"
                        )
                    else:
                        self.logger.error("Loaded model is not fitted")
                        self.pipeline = None

                except Exception as e:
                    self.logger.error(f"Failed to load model: {str(e)}")
                    self.logger.error(traceback.format_exc())
                    self.pipeline = None

        if self.pipeline is None and not train_mode:
            self.logger.warning(
                "No pre-trained model found, creating new pipeline"
            )
            self.pipeline = self.create_pipeline()

        if train_mode:
            if not db_config:
                raise ValueError("db_config required when train_mode=True")
            
            try:
                if len(available_models) > 0:
                    self.model_version = self._increment_version()
                metrics = self.train_from_mongodb(**db_config)
                self.model_version = metrics.get(
                    'model_version', self.model_version
                )

            except Exception as e:
                self.logger.error(f"Initial training failed: {str(e)}")
                raise

    def is_ready(self: Self):
        """Check if predictor is ready for predictions."""

        return hasattr(self, 'pipeline') and self.pipeline is not None

    def train_from_mongodb(
        self: Self,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> dict[str, object]:
        """Train and version a new model from MongoDB data."""

        try:
            db = current_app.db
            data = pd.DataFrame(list(db.applications.find(
                {
                    "vacancy_description": {"$exists": True, "$ne": ""},
                    "ml.success_probability": {"$exists": True, "$ne": None}
                },
                {
                    "vacancy_description": 1,
                    "role": 1,
                    "source": 1,
                    "ml.success_probability": 1,
                    "_id": 0
                }
            )))
            
            self._validate_training_data(data, 'ml.success_probability')
            
            X = self.feature_processor(data)
            y = data['ml.success_probability'].astype(float)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state,
                stratify=pd.cut(y, bins=[0, 0.3, 0.7, 1.0])
            )

            self.pipeline = self.create_pipeline()
            self.pipeline.fit(X_train, y_train)
            
            model_path = self._save_model()
            
            y_pred_proba = self.pipeline.predict_proba(X_test)[:, 1]
            metrics = {
                "model_version": self.model_version,
                "train_accuracy": float(self.pipeline.score(X_train, y_train)),
                "test_accuracy": float(self.pipeline.score(X_test, y_test)),
                "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
                "feature_count": X.shape[1],
                "samples": {
                    "total": len(data),
                    "train": len(X_train),
                    "test": len(X_test),
                    "cats": self._get_class_distribution(y_train, y_test)
                },
                "model_path": model_path
            }

            current_app.logger.info(
                f"Training successful - v{self.model_version}\n"
                f"Test Accuracy: {metrics['test_accuracy']:.2f}, "
                f"AUC: {metrics['roc_auc']:.2f}"
            )

            return metrics
            
        except ValueError as e:
            current_app.logger.error(f"Data validation failed: {str(e)}")
            raise

        except Exception as e:
            current_app.logger.error(
                f"Training failed: {str(e)}\n{traceback.format_exc()}"
            )
            if hasattr(self, 'pipeline'):
                del self.pipeline
            raise RuntimeError(f"Training aborted: {str(e)}") from e

    def predict(
        self: Self,
        job_data: dict,
        german_level: str = None
    ) -> float:
        """Predict job application success probability."""

        REQUIRED_KEYS = {'vacancy_description', 'role', 'source'}
        PROFICIENCY_ORDER = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']

        if not self._is_fitted():
            raise RuntimeError(
                f"Pipeline is not fitted yet. "
                f"Please train the model first."
            )

        if not all(key in job_data for key in REQUIRED_KEYS):
            missing = REQUIRED_KEYS - set(job_data.keys())
            raise ValueError(f"Missing required keys: {missing}")

        try:
            features = self.feature_processor(pd.DataFrame([job_data]))
            
            if features.iloc[0]['lang_german_required']:
                if not german_level:
                    return 0.0

                try:
                    required_idx = PROFICIENCY_ORDER.index(
                        features.iloc[0]['lang_german_level']
                    )
                    user_idx = PROFICIENCY_ORDER.index(german_level)
                    if required_idx > user_idx:
                        return 0.0

                except ValueError:
                    return 0.0

            raw_prediction = self.pipeline.predict(features)

            # Handle different prediction output formats            
            if (
                hasattr(raw_prediction, 'shape') and 
                len(raw_prediction.shape) > 1
            ):
                prediction = float(raw_prediction[0][1])
            elif (
                hasattr(raw_prediction, '__len__') and 
                len(raw_prediction) > 0
            ):
                prediction = float(raw_prediction[0])
            else:
                prediction = float(raw_prediction)
            
            return max(0.0, min(1.0, prediction))

        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}") from e

    def predict_batch(
        self: Self,
        query_filter: Optional[dict] = None,
        update_threshold: float = 0.01,
        max_documents: Optional[int] = None
    ) -> dict:
        """Batch predict success probabilities and update MongoDB."""

        if not current_app or not hasattr(current_app, 'db'):
            raise RuntimeError(
                "Flask application context with DB not available"
            )

        db = current_app.db
        stats = {
            'processed': 0,
            'updated': 0,
            'skipped': 0,
            'errors': 0,
            'error_details': []
        }

        base_query = {
            "vacancy_description": {"$exists": True, "$ne": ""},
            "role": {"$exists": True},
            "source": {"$exists": True}
        }

        # Merge with user-provided filter
        final_query = {**base_query, **(query_filter or {})}

        try:
            total_count = db.applications.count_documents(final_query)
            if max_documents:
                total_count = min(total_count, max_documents)
            
            current_app.logger.info(
                f"Starting batch prediction on {total_count} documents"
            )

            cursor = db.applications.find(final_query)
            if max_documents:
                cursor.limit(max_documents)

            for document in cursor:
                stats['processed'] += 1
                doc_id = document['_id']

                if stats['processed'] % 100 == 0:
                    current_app.logger.info(
                        f"Processed {stats['processed']}"
                        f"/{total_count} documents"
                    )

                try:
                    job_data = {
                        'vacancy_description': (
                            document.get('vacancy_description', '')
                        ),
                        'role': document.get('role', ''),
                        'source': document.get('source', ''),
                        'tech_stack': document.get('tech_stack', []),
                        'industry': document.get('industry', ''),
                        'company': document.get('company', ''),
                        'location': document.get('location', ''),
                        'experience_level': (
                            document.get('experience_level', '')
                        ),
                    }

                    ml_data = document.get('ml', {})
                    current_pred = ml_data.get('success_probability')
                    current_german = ml_data.get('german_level')
                    current_pred_date = ml_data.get('last_prediction_date')

                    german_level = (
                        document.get('requirements', {}).get('german_level')
                    )

                    prediction = self.predict(
                        job_data, german_level=german_level
                    )

                    needs_update = (
                        current_pred is None or
                        abs(prediction - current_pred) > update_threshold or
                        current_german != german_level or
                        (current_pred_date and 
                        (datetime.now() - current_pred_date).days > 30)
                    )

                    if needs_update:
                        update_data = {
                            "ml.success_probability": prediction,
                            "ml.german_level": german_level,
                            "ml.last_prediction_date": datetime.now(),
                            "ml.model_version": self.model_version
                        }

                        result = db.applications.update_one(
                            {"_id": doc_id},
                            {"$set": update_data}
                        )
                        
                        if result.modified_count > 0:
                            stats['updated'] += 1
                        else:
                            stats['skipped'] += 1
                    else:
                        stats['skipped'] += 1

                except Exception as e:
                    stats['errors'] += 1
                    stats['error_details'].append({
                        "document_id": str(doc_id),
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    })
                    current_app.logger.error(
                        f"Batch prediction failed for "
                        f"document {doc_id}: {str(e)}"
                    )
                    continue

            current_app.logger.info(
                f"Batch prediction completed: "
                f"{stats['processed']} processed, "
                f"{stats['updated']} updated, "
                f"{stats['skipped']} skipped, "
                f"{stats['errors']} errors"
            )

            return stats

        except Exception as e:
            current_app.logger.error(
                f"Batch prediction failed completely: {str(e)}"
            )
            raise RuntimeError(
                f"Batch prediction failed: {str(e)}"
            ) from e

    def create_pipeline():
        """Factory for creating new model pipelines."""

        preprocessor = make_column_transformer(
            (TechStackTransformer(), ['tech_stack']),
            (OneHotEncoder(handle_unknown='ignore'), ['industry']),
            (OneHotEncoder(handle_unknown='ignore'), ['source']),
            (OneHotEncoder(handle_unknown='ignore'), ['german_level']),
            ('passthrough', ['relative_seniority']),
            (LanguageAwareTfidf(max_features=50), ['processed_text', 'description_lang']),
            remainder='drop'
        )

        return make_pipeline(
            preprocessor,
            GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=4,
                loss='squared_error'
            )
        )

    def get_model_info(self: Self) -> dict:
        """Get model metadata with pipeline configuration."""

        if not hasattr(self, 'pipeline') or not self.pipeline:
            raise RuntimeError("Model pipeline not initialized")

        info = {
            "version": getattr(self, 'model_version', 'unversioned'),
            "last_retrained": self._get_model_timestamp(),
            "model_type": self.pipeline[-1].__class__.__name__,
            "feature_count": 0,
            "pipeline_steps": []
        }

        try:
            ct = self.pipeline.named_steps.get('columntransformer')
            if ct:
                info['features'] = list(ct.get_feature_names_out())
                info['feature_count'] = len(info['features'])
                
                info['pipeline_steps'] = [
                    {
                        'transformer': str(transformer.__class__.__name__),
                        'columns': columns,
                        'params': transformer.get_params()
                    }
                    for transformer, columns, _ in ct.transformers_
                ]

            gb = self.pipeline.named_steps.get('gradientboostingregressor')
            if gb:
                info['regressor_params'] = {
                    'n_estimators': gb.n_estimators,
                    'learning_rate': gb.learning_rate,
                    'max_depth': gb.max_depth,
                    'class_weight': str(gb.class_weight)
                }

        except Exception as e:
            current_app.logger.error(
                f"Failed to extract pipeline info: {str(e)}"
            )
            info['error'] = f"Metadata incomplete: {str(e)}"

        if hasattr(self, 'training_metrics'):
            info.update({
                k: v for k, v in self.training_metrics.items()
                if not k.startswith('_')
            })

        return info

    def _is_fitted(self: Self) -> bool:
        """Check if the pipeline is fitted."""

        if self.pipeline is None:
            return False
        
        try:
            if (
                hasattr(self.pipeline, 'steps') and 
                len(self.pipeline.steps) > 0
            ):
                first_step = self.pipeline.steps[0][1]
                if hasattr(first_step, 'transform'):
                    return True
            return False

        except:
            return False

    def _get_available_models(self: Self) -> list[str]:
        """Return list of available model files."""

        try:
            return [
                os.path.join(self._model_dir, f)
                for f in os.listdir(self._model_dir) 
                if f.endswith('.pkl')
            ]

        except FileNotFoundError:
            return []

    def _get_class_distribution(
        self: Self,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> dict[str, dict[float, float]]:
        """Calculate class distribution for train/test sets."""

        def _calculate_dist(y: pd.Series) -> dict[float, float]:
            if hasattr(y, 'value_counts'):
                return y.value_counts(normalize=True).round(2).to_dict()
            return {}

        return {
            "train": _calculate_dist(y_train),
            "test": _calculate_dist(y_test)
        }

    def _validate_training_data(
        self: Self,
        data: pd.DataFrame,
        target_field: str
    ) -> None:
        """Validate training data meets requirements."""

        if len(data) < 100:
            raise ValueError(
                f"Insufficient samples ({len(data)}). Need ≥100 records"
            )
        
        if target_field not in data.columns:
            raise ValueError(
                f"Target field '{target_field}' not found in data"
            )
        
        if data[target_field].isna().any():
            raise ValueError(
                f"Null values found in target field '{target_field}'"
            )
        
        if not all(0 <= val <= 1 for val in data[target_field].dropna()):
            raise ValueError(
                f"Target values in '{target_field}' must be between 0.0 and 1.0"
            )
        
        required_features = ['vacancy_description', 'role', 'source']
        missing = [f for f in required_features if f not in data.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        
        if any(not isinstance(desc, str) or len(desc) < 50 
        for desc in data['vacancy_description']):
            raise ValueError(
                "All vacancy descriptions must be strings ≥50 characters"
            )

    def _load_model(self: Self, path: str) -> Pipeline:
        """Load serialized model pipeline."""

        # Ensure the class is registered every time we load
        import __main__
        __main__.TechStackTransformer = TechStackTransformer
        __main__.LanguageAwareTfidf = LanguageAwareTfidf

        current_module = sys.modules[__name__]
        setattr(
            current_module, 'TechStackTransformer', TechStackTransformer
        )
        setattr(
            current_module, 'LanguageAwareTfidf', LanguageAwareTfidf
        )
        
        self.logger.info(f"Loading model from: {path}")

        try:
            pipeline = joblib.load(path)
            self.logger.info("Model loaded successfully")
            
            if hasattr(pipeline, 'steps') and len(pipeline.steps) > 0:
                first_step = pipeline.steps[0][1]
                if hasattr(first_step, 'transform'):
                    self.logger.info("Pipeline appears to be fitted")
                else:
                    self.logger.warning(
                        "First step doesn't have transform method"
                    )
            else:
                self.logger.warning("Pipeline has no steps")
                
            return pipeline
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {str(e)}")
            raise
        
    def _get_latest_model(self: Self) -> Optional[str]:
        """Return the path to the latest model."""

        model_files = self._get_available_models()
        if not model_files:
            return None

        enhanced_model_path = os.path.join(
            self._model_dir, "enhanced_model.pkl"
        )
        if os.path.exists(enhanced_model_path):
            return enhanced_model_path

        versioned_files = [
            f for f in model_files 
            if (
                '_v' in os.path.basename(f) and 
                not os.path.basename(f).startswith('enhanced_model')
            )
        ]

        if versioned_files:
            versioned_files.sort(
                key=lambda f: os.path.getmtime(f), reverse=True
            )
            latest_versioned = versioned_files[0]
            
            version_match = re.search(
                r'_v([\d.]+)_', os.path.basename(latest_versioned)
            )
            if version_match:
                self.model_version = version_match.group(1)
            return latest_versioned

        model_files.sort(key=lambda f: os.path.getmtime(f), reverse=True)
        return model_files[0]

    def _increment_version(
            self: Self,
            breaking_change: bool = False
    ) -> str:
        """Generate next semantic version."""

        try:
            current_major = getattr(self, 'major_version', None)
            
            if current_major is not None and breaking_change:
                self.major_version += 1
                return f"{self.major_version}.0"
            
            major, minor = map(int, self.model_version.split('.'))
            
            if breaking_change:
                return f"{major + 1}.0"
            else:
                return f"{major}.{minor + 1}"
                
        except (ValueError, AttributeError, IndexError) as e:
            error_msg = (
                f"Version increment failed ({str(e)}). "
                f"Defaulting to 1.0"
            )
            self.logger.warning(error_msg)
            return "1.0"

    def _save_model(self: Self) -> str:
        """Save pipeline with versioned filename."""

        os.makedirs(self._model_dir, exist_ok=True)

        model_path = os.path.join(
            self._model_dir,
            f"model_v{self.model_version}_{datetime.now().strftime('%Y%m%d')}.pkl"
        )
        joblib.dump(self.pipeline, model_path)

        # Update latest reference
        with open(os.path.join(self._model_dir, 'latest_model.txt'), 'w') as f:
            f.write(os.path.basename(model_path))
        
        return model_path


    def _get_model_timestamp(self: Self) -> str:
        """Get last retrained timestamp safely."""

        model_path = os.path.join('models', 'enhanced_model.pkl')
        try:
            if os.path.exists(model_path):
                return datetime.fromtimestamp(
                    os.path.getmtime(model_path)
                ).isoformat()

        except OSError:
            pass

        return "unknown"

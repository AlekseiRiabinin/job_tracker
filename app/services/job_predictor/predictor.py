import os
import joblib
import pandas as pd
from datetime import datetime
from typing import Self, Optional
from sklearn.pipeline import Pipeline ,make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
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
        db_config: Optional[dict] = None
    ) -> None:
        """Initialize prediction service."""
        self.model_version = "1.0"
        self.feature_processor = create_enhanced_features

        if train_mode and db_config:
            self.train_from_mongodb(**db_config)
        else:
            self.pipeline = self._load_model(
                model_path or 'models/enhanced_model.pkl'
            )

    def _load_model(self: Self, path: str) -> Pipeline:
        """Load serialized model pipeline."""
        return joblib.load(path)

    def train_from_mongodb(
        self: Self,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> None:
        """Train model using data from MongoDB collection."""
        try:
            from flask import current_app
            
            db = current_app.db
            target_field = 'ml.success_probability'
            model_dir = current_app.config['ML_MODEL_DIR']
            
            data = pd.DataFrame(list(db.applications.find(
                {
                    "vacancy_description": {"$exists": True},
                    target_field: {"$exists": True}
                },
                {
                    "vacancy_description": 1,
                    "role": 1,
                    "source": 1,
                    "ml": 1,
                    "_id": 0
                }
            )))
            
            self._validate_training_data(data, target_field)
            
            X = self.feature_processor(data)
            y = data[target_field].astype(float)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state,
                stratify=y
            )
            
            self.pipeline = create_pipeline()
            self.pipeline.fit(X_train, y_train)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(
                model_dir,
                f"model_v{self.model_version}_{timestamp}.pkl"
            )
            
            os.makedirs(model_dir, exist_ok=True)
            joblib.dump(self.pipeline, model_path)
            
            latest_path = os.path.join(model_dir, "latest_model.txt")
            with open(latest_path, 'w') as f:
                f.write(os.path.basename(model_path))
            
            metrics = {
                "train_accuracy": float(self.pipeline.score(X_train, y_train)),
                "test_accuracy": float(self.pipeline.score(X_test, y_test)),
                "model_path": model_path,
                "samples": {
                    "total": len(data),
                    "train": len(X_train),
                    "test": len(X_test)
                }
            }
            
            current_app.logger.info(
                f"Training complete. "
                f"Train accuracy: {metrics['train_accuracy']:.2f}, "
                f"Test accuracy: {metrics['test_accuracy']:.2f}"
            )
            
            return metrics
            
        except Exception as e:
            current_app.logger.error(f"Training failed: {str(e)}")
            raise RuntimeError(f"Training failed: {str(e)}") from e

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
                f"Target field '{target_field}' not found"
            )
        if data[target_field].isna().any():
            raise ValueError(
                f"Null values found in target field"
            )
        if not all(
            0 <= val <= 1 
            for val in data[target_field].dropna()
        ):
            raise ValueError(
                "Target values must be between 0.0 and 1.0"
            )

        required_features = ['vacancy_description', 'role', 'source']
        missing = [
            f for f in required_features 
            if f not in data.columns
        ]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

    def predict(
        self: Self,
        job_data: dict,
        german_level: str = None
    ) -> float:
        """Predict job application success probability."""
        REQUIRED_KEYS = {'vacancy_description', 'role', 'source'}

        if not all(key in job_data for key in REQUIRED_KEYS):
            missing = REQUIRED_KEYS - set(job_data.keys())
            raise ValueError(f"Missing required keys: {missing}")

        try:
            features = self.feature_processor(pd.DataFrame([job_data]))

            if (features.iloc[0]['german_required'] and (
                not german_level 
                or features.iloc[0]['german_level'] > german_level
            )):
                return 0.0

            return float(self.pipeline.predict_proba(features)[0][1])

        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def get_model_info(self: Self) -> dict:
        """Get comprehensive model metadata."""
        if not hasattr(self, 'pipeline'):
            raise RuntimeError("Model not loaded or trained")
        
        model_path = 'models/enhanced_model.pkl'
        last_retrained = "Unknown"
        if os.path.exists(model_path):
            last_retrained = datetime.fromtimestamp(
                os.path.getmtime(model_path)
                .strftime('%Y-%m-%d %H:%M:%S')
            )
        
        feature_names = []
        try:
            ct = self.pipeline.named_steps.get('columntransformer')
            if ct:
                feature_names = list(ct.get_feature_names_out())
        except Exception as e:
            feature_names = [f"Error: {str(e)}"]

        model_stats = {}
        if hasattr(self, 'training_metrics'):
            model_stats.update(self.training_metrics)
        
        return {
            "version": self.model_version,
            "last_retrained": last_retrained,
            "features": feature_names,
            "model_type": str(self.pipeline[-1].__class__.__name__),
            "feature_count": len(feature_names),
            **model_stats
        }


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
        GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=4,
            class_weight='balanced'
        )
    )

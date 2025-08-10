import joblib
import pandas as pd
from typing import Self, Optional, Any
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from pymongo import MongoClient
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

    def _load_model(self: Self, path: str) -> Any:
        """Load serialized model pipeline."""
        return joblib.load(path)

    def train_from_mongodb(
        self: Self,
        db_uri: str,
        db_name: str, 
        collection: str,
        target_field: str = 'success_probability',
        test_size: float = 0.2,
        random_state: int = 42
    ) -> None:
        """Train model using data from MongoDB."""
        client = MongoClient(db_uri)
        db = client[db_name]
        collection = db[collection]

        data = pd.DataFrame(list(collection.find()))
        X = self.feature_processor(data)
        y = data[target_field]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        self.pipeline = create_pipeline()
        self.pipeline.fit(X_train, y_train)
        
        joblib.dump(self.pipeline, 'models/enhanced_model.pkl')

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
            
            if (features.iloc[0]['german_required'] and 
                (not german_level or 
                 features.iloc[0]['german_level'] > german_level)):
                return 0.0
            
            return float(self.pipeline.predict_proba(features)[0][1])
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def get_model_info(self: Self) -> dict:
        """Get model metadata."""
        return {
            "version": self.model_version,
            "last_retrained": "2023-07-20",
            "features": list(
                self.pipeline
                    .named_steps['columntransformer']
                    .get_feature_names_out()
            )
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

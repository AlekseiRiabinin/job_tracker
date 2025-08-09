import joblib
import pandas as pd
from typing import Self
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder
from .feature_engine import (
    TechStackTransformer,
    LanguageAwareTfidf,
    create_enhanced_features
)


class JobPredictor:
    def __init__(self: Self, model_path: str = None) -> None:
        """Initialize prediction service."""
        self.pipeline = self._load_model(model_path or 'models/enhanced_model.pkl')
        self.feature_processor = create_enhanced_features

    def _load_model(self, path: str):
        """Load serialized model pipeline."""
        return joblib.load(path)

    def predict(self, job_data: dict, german_level: str = None) -> float:
        """Predict job application success probability."""
        features = self.feature_processor(pd.DataFrame([job_data]))
        
        if (features.iloc[0]['german_required'] and 
            (not german_level or 
             features.iloc[0]['german_level'] > german_level)):
            return 0.0
        
        return self.pipeline.predict_proba(features)[0][1]



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

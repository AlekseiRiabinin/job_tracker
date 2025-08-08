import re
import spacy
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Any, Self
from langdetect import detect
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer


# Initialize language models
nlp_de = spacy.load("de_core_news_sm")
nlp_en = spacy.load("en_core_web_sm")


class EnhancedFeatureExtractor:
    """Multilingual feature extraction."""

    @staticmethod
    def detect_language(text: str) -> str:
        try:
            lang = detect(text[:500])
            return 'de' if lang == 'de' else 'en'
        except:
            return 'en'

    @staticmethod
    def extract_tech_stack(description: str) -> dict[str, int]:
        """Find tech stack keywords."""
        cv_tech = {
            'python': r'\bpython\b',
            'scala': r'\bscala\b',
            'spark': r'\bspark\b|\bpyspark\b',
            'kafka': r'\bkafka\b',
            'airflow': r'\bairflow\b',
            'mongodb': r'\bmongo(db)?\b',
            'flask': r'\bflask\b',
            'sql': r'\bsql\b|\bpostgresql\b|\bpl/pgsql\b',
            'docker': r'\bdocker\b',
            'tensorflow': r'\btensorflow\b',
            'fastapi': r'\bfastapi\b'
        }
        return {
            tech: int(bool(re.search(pattern, description.lower())))
            for tech, pattern in cv_tech.items()
        }

    @staticmethod
    def extract_language_requirements(
        description: str
    ) -> dict[str, bool | str | None]:
        """Detects language requirements in job descriptions."""

        german_main_pattern = re.compile(
            r'(?:fließend|verhandlungssicher|gut|geschäftssicher)\s+deutsch|'
            r'deutsch\s*(?:kenntnisse|erforderlich|voraussetzung|kenntnisse)',
            re.IGNORECASE
        )

        english_pattern = re.compile(
            r'english\s+(?:fluent|proficient|required|working\s+knowledge)',
            re.IGNORECASE
        )

        german_level_patterns = {
            'A1': re.compile(
                r'A1\s*deutsch|'
                r'grundkenntnisse|'
                r'anfängerkenntnisse',
                re.IGNORECASE
            ),
            'A2': re.compile(
                r'A2\s*deutsch|'
                r'basiskenntnisse|'
                r'einfache\s+konversation',
                re.IGNORECASE
            ),
            'B1': re.compile(
                r'B1\s*deutsch|'
                r'fortgeschrittene\s+kenntnisse|'
                r'selbständige\s+sprachverwendung',
                re.IGNORECASE
            ),
            'B2': re.compile(
                r'B2\s*deutsch|'
                r'gute\s+kenntnisse|'
                r'berufsbezogene\s+sprachkenntnisse',
                re.IGNORECASE
            ),
            'C1': re.compile(
                r'C1\s*deutsch|'
                r'verhandlungssicher|'
                r'fließend|'
                r'geschäftssicher',
                re.IGNORECASE
            ),
            'C2': re.compile(
                r'C2\s*deutsch|'
                r'muttersprachlich|'
                r'herausragende\s+kenntnisse',
                re.IGNORECASE
            )
        }

        requirements = {
            'german_required': bool(german_main_pattern.search(description)),
            'english_required': bool(english_pattern.search(description)),
            'german_level': None
        }

        if requirements['german_required']:
            for level in ['C2', 'C1', 'B2', 'B1', 'A2', 'A1']:
                if german_level_patterns[level].search(description):
                    requirements['german_level'] = level
                    break

        return requirements

    @staticmethod
    def extract_industry_focus(description: str, lang: str = 'en') -> str:
        """Detects industry focus from job description."""
        industry_patterns = {
            'en': {
                'finance': (
                    r'\b(financ|bank|invest|accounting|'
                    r'audit|tax|trad|republic|brokerage)\w*\b'
                ),
                'healthcare': (
                    r'\b(health|medical|pharma|'
                    r'hospital|clinic|patient)\w*\b'
                ),
                'tech': (
                    r'\b(tech|software|it|computer|system|'
                    r'developer|engineer|ai|artificial intelligence|'
                    r'deep learning|machine learning|cloud|data science)\w*\b'
                ),
                'manufacturing': (
                    r'\b(manufactur|production|factory|'
                    r'plant|assembly|automotive|car|vehicle)\w*\b'
                ),
                'retail': (
                    r'\b(retail|store|shop|'
                    r'e.?commerce|merchandis)\w*\b'
                ),
                'logistics': (
                    r'\b(logistics|delivery|'
                    r'supply chain|transport|shipping)\w*\b'
                ),
                'food': r'\b(food|grocery|restaurant|meal)\w*\b',
                'telecom': r'\b(telecom|telecommunication|mobile network)\w*\b',
                'energy': r'\b(energy|solar|renewable|power|electric)\w*\b',
                'aerospace': r'\b(aerospace|aviation|space|satellite)\w*\b',
                'consulting': r'\b(consult|advisory)\w*\b',
                'insurance': r'\b(insurance|actuar)\w*\b',
                'gaming': r'\b(game|gaming)\w*\b',
                'social': r'\b(social media|social network)\w*\b',
                'semiconductor': r'\b(semiconductor|chip|microelectronic)\w*\b'
            },
            'de': {
                'finance': (
                    r'\b(finan|bank|invest|buchhalt|'
                    r'rechnungs|steuer|handel)\w*\b'
                ),
                'healthcare': (
                    r'\b(gesundheit|medizin|pharma|'
                    r'krankenhaus|klinik|patient)\w*\b'
                ),
                'tech': (
                    r'\b(tech|software|it|computer|system|'
                    r'entwickler|ingenieur|ki|künstliche intelligenz|'
                    r'maschinelles lernen|datenwissenschaft)\w*\b'
                ),
                'manufacturing': (
                    r'\b(produktion|fabrik|werk|'
                    r'montage|herstellung|auto|fahrzeug)\w*\b'
                ),
                'retail': (
                    r'\b(einzelhandel|laden|'
                    r'geschäft|e.?commerce|handel)\w*\b'
                ),
                'logistics': r'\b(logistik|lieferung|transport)\w*\b',
                'food': r'\b(lebensmittel|nahrung|restaurant|mahlzeit)\w*\b',
                'telecom': r'\b(telekommunikation|mobilfunk)\w*\b',
                'energy': r'\b(energie|solar|erneuerbar|strom)\w*\b',
                'aerospace': r'\b(luftfahrt|raumfahrt|satellit)\w*\b',
                'consulting': r'\b(beratung|berater)\w*\b',
                'insurance': r'\b(versicherung)\w*\b',
                'gaming': r'\b(spiel|gaming)\w*\b',
                'social': r'\b(soziale medien)\w*\b',
                'semiconductor': r'\b(hälbleiter|chip|mikroelektronik)\w*\b'
            }
        }

        description_lower = description.lower()
        patterns = industry_patterns.get(lang, industry_patterns['en'])

        for industry, pattern in patterns.items():
            if re.search(pattern, description_lower, re.IGNORECASE):
                return industry
        return 'other'

    @staticmethod
    def preprocess_text(text: str, lang: str) -> str:
        """Language-specific preprocessing."""
        doc = nlp_de(text) if lang == 'de' else nlp_en(text)
        return " ".join([
            token.lemma_.lower() for token in doc 
            if not token.is_stop and token.is_alpha
        ])


def create_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering with multilingual support."""

    df['description_lang'] = df['vacancy_description'].apply(
        EnhancedFeatureExtractor.detect_language
    )
    df['tech_stack'] = df['vacancy_description'].apply(
        EnhancedFeatureExtractor.extract_tech_stack
    )
    df['language_reqs'] = df['vacancy_description'].apply(
        EnhancedFeatureExtractor.extract_language_requirements
    )
    df['industry'] = df['vacancy_description'].apply(
        lambda x: EnhancedFeatureExtractor.extract_industry_focus(
            x, 
            lang=EnhancedFeatureExtractor.detect_language(x)
        )
    )

    # Expand nested features
    lang_reqs = pd.json_normalize(df['language_reqs'])
    df = pd.concat([df, lang_reqs], axis=1)

    # Add relative seniority
    df['relative_seniority'] = df['role'].apply(
        lambda x: 1 if 'senior' in x.lower() else 
                 -1 if 'junior' in x.lower() else 0
    )

    # Preprocess text by language
    df['processed_text'] = df.apply(
        lambda x: EnhancedFeatureExtractor.preprocess_text(
            x['vacancy_description'],
            x['description_lang']
        ), 
        axis=1
    )
    return df


class TechStackTransformer(BaseEstimator, TransformerMixin):
    """Transforms a column of tech stack dicts into a feature matrix."""

    def __init__(self: Self) -> None:
        """Initialize the transformer."""
        self.feature_names_: Optional[list[str]] = None

    def fit(self: Self, X: pd.DataFrame) -> Self:
        """Learn the feature names from the tech stack dictionaries."""
        all_techs = set()
        for tech_dict in X['tech_stack']:
            if isinstance(tech_dict, dict):
                all_techs.update(tech_dict.keys())
        self.feature_names_ = sorted(all_techs)
        return self

    def transform(self: Self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform tech stack dicts into a binary feature matrix."""
        if self.feature_names_ is None:
            raise ValueError("Must call fit() before transform()")

        # Create a DataFrame with 0/1 values for each tech
        transformed = []
        for tech_dict in X['tech_stack']:
            row = {tech: 0 for tech in self.feature_names_}
            if isinstance(tech_dict, dict):
                row.update({k: 1 for k, v in tech_dict.items() if v and k in row})
            transformed.append(row)

        return pd.DataFrame(transformed, columns=self.feature_names_)

    def get_feature_names_out(self: Self) -> list[str]:
        """Get output feature names for transformation."""
        if self.feature_names_ is None:
            raise ValueError("Transformer not fitted yet")
        return self.feature_names_


class LanguageAwareTfidf(BaseEstimator, TransformerMixin):
    """Language-aware TF-IDF vectorizer."""

    def __init__(self: Self, max_features: int = 50) -> None:
        """Initialize the language-aware vectorizer."""
        self.max_features = max_features

        self.vectorizer_en = TfidfVectorizer(
            max_features=max_features,
            stop_words=spacy.lang.en.stop_words.STOP_WORDS
        )
        self.vectorizer_de = TfidfVectorizer(
            max_features=max_features,
            stop_words=spacy.lang.de.stop_words.STOP_WORDS
        )

    def fit(self: Self, X: pd.DataFrame) -> Self:
        """Fit the vectorizers to English and German text subsets."""

        texts_en = X[X['description_lang'] == 'en']['processed_text']
        texts_de = X[X['description_lang'] == 'de']['processed_text']

        self.vectorizer_en.fit(texts_en)
        self.vectorizer_de.fit(texts_de)
        return self

    def transform(self: Self, X: pd.DataFrame) -> np.ndarray | csr_matrix:
        """Transform text data into a combined TF-IDF feature matrix."""
        if not all(
            col in X.columns 
            for col in ['processed_text', 'description_lang']
        ):
            raise ValueError(
                "Input DataFrame must contain 'processed_text' "
                "and 'description_lang' columns"
            )

        texts_en = X[X['description_lang'] == 'en']['processed_text']
        texts_de = X[X['description_lang'] == 'de']['processed_text']

        tfidf_en = self.vectorizer_en.transform(texts_en)
        tfidf_de = self.vectorizer_de.transform(texts_de)

        # Combine results while preserving original row order
        tfidf_all = np.zeros((len(X), self.max_features))
        en_mask = X['description_lang'] == 'en'
        de_mask = X['description_lang'] == 'de'

        tfidf_all[en_mask] = tfidf_en.toarray()
        tfidf_all[de_mask] = tfidf_de.toarray()

        return (
            csr_matrix(tfidf_all) 
            if isinstance(tfidf_en, csr_matrix) 
            else tfidf_all
        )

    def get_feature_names_out(self: Self) -> list[str]:
        """Get output feature names with language prefixes."""
        en_features = [
            f"en_{f}" 
            for f in self.vectorizer_en.get_feature_names_out()
        ]
        de_features = [
            f"de_{f}" 
            for f in self.vectorizer_de.get_feature_names_out()
        ]
        return en_features + de_features


# Revised pipeline
preprocessor = make_column_transformer(
    (TechStackTransformer(), ['tech_stack']),
    (OneHotEncoder(handle_unknown='ignore'), ['industry']),
    (OneHotEncoder(handle_unknown='ignore'), ['source']),
    (OneHotEncoder(handle_unknown='ignore'), ['german_level']),
    ('passthrough', ['relative_seniority']),
    (LanguageAwareTfidf(max_features=50), ['processed_text', 'description_lang']),
    remainder='drop'
)

pipeline = make_pipeline(
    preprocessor,
    GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=4,
        class_weight='balanced'
    )
)


def predict_job_success(job_post: dict, german_level: str = None) -> float:
    """Predict job success probability with language awareness."""
    # Load model if not done globally
    if not hasattr(predict_job_success, 'pipeline'):
        predict_job_success.pipeline = joblib.load('enhanced_model.pkl')
    
    # Create features
    features = create_enhanced_features(pd.DataFrame([job_post]))
    
    # Hard filter for German requirements
    if (features.iloc[0]['german_required'] and 
        (not german_level or 
         features.iloc[0]['german_level'] > german_level)):
        return 0.0
    
    # Predict
    proba = predict_job_success.pipeline.predict_proba(features)[0][1]
    return proba

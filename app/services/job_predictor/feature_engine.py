import re
import spacy
import pandas as pd
from langdetect import detect
from typing import Optional, Literal, Pattern, Self
from scipy.sparse import csr_matrix, vstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en.stop_words import STOP_WORDS as EN_STOP_WORDS
from spacy.lang.de.stop_words import STOP_WORDS as DE_STOP_WORDS
from .config import (
    LanguagePatterns,
    TECH_KEYWORDS,
    INDUSTRY_PATTERNS,
    LANGUAGE_PATTERNS,
    SENIORITY_PATTERNS
)


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
        return {
            tech: int(bool(re.search(pattern, description.lower())))
            for tech, pattern in TECH_KEYWORDS.items()
        }

    @staticmethod
    def extract_language_requirements(
        description: str
    ) -> LanguagePatterns:
        """Detects language requirements in job descriptions."""
        requirements = {
            'german_required': bool(
                LANGUAGE_PATTERNS['german'].search(description)
            ),
            'english_required': bool(
                LANGUAGE_PATTERNS['english'].search(description)
            ),
            'german_level': None
        }

        if requirements['german_required']:
            german_levels = LANGUAGE_PATTERNS['german_levels']
            for level, pattern in german_levels.items():
                pattern: Pattern[str] = german_levels[level]
                if pattern.search(description):
                    requirements['german_level'] = level
                break

        return requirements

    @staticmethod
    def extract_industry_focus(description: str, lang: str = 'en') -> str:
        """Detects industry focus from job description."""
        description_lower = description.lower()
        patterns = INDUSTRY_PATTERNS.get(lang, INDUSTRY_PATTERNS['en'])

        for industry, pattern in patterns.items():
            if re.search(pattern, description_lower, re.IGNORECASE):
                return industry
        return 'other'

    @staticmethod
    def calculate_seniority(role: str) -> Literal[-1, 0, 1]:
        """Determine seniority level using configured patterns."""
        role_lower = role.lower()
        if SENIORITY_PATTERNS['senior'].search(role_lower):
            return 1
        elif SENIORITY_PATTERNS['junior'].search(role_lower):
            return -1
        return 0

    @staticmethod
    def preprocess_text(text: str, lang: str) -> str:
        """Language-specific preprocessing."""
        doc = nlp_de(text) if lang == 'de' else nlp_en(text)
        return " ".join([
            token.lemma_.lower() for token in doc 
            if not token.is_stop and token.is_alpha
        ])


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
            stop_words=EN_STOP_WORDS
        )
        self.vectorizer_de = TfidfVectorizer(
            max_features=max_features,
            stop_words=DE_STOP_WORDS
        )

    def fit(self: Self, X: pd.DataFrame) -> Self:
        """Fit the vectorizers to English and German text subsets."""

        if not {'processed_text', 'description_lang'}.issubset(X.columns):
            raise ValueError(
                f"Input DataFrame must contain 'processed_text' "
                f"and 'description_lang' columns"
            )

        texts_en = X[X['description_lang'] == 'en']['processed_text']
        texts_de = X[X['description_lang'] == 'de']['processed_text']

        self.vectorizer_en.fit(texts_en)
        self.vectorizer_de.fit(texts_de)

        return self

    def transform(self: Self, X: pd.DataFrame) -> csr_matrix:
        """Transform text data into a combined TF-IDF feature matrix."""
        results = []
        for _, row in X.iterrows():
            if row['description_lang'] == 'en':
                results.append(
                    self.vectorizer_en.transform([row['processed_text']])
                )
            else:
                results.append(
                    self.vectorizer_de.transform([row['processed_text']])
                )

        return vstack(results)

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


def create_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering pipeline with multilingual support."""
    df = df.copy()
    
    lang_detection = df['vacancy_description'].apply(
        EnhancedFeatureExtractor.detect_language
    )
    df['description_lang'] = lang_detection
    
    df['tech_stack'] = df['vacancy_description'].apply(
        EnhancedFeatureExtractor.extract_tech_stack
    )
    df['language_reqs'] = df['vacancy_description'].apply(
        EnhancedFeatureExtractor.extract_language_requirements
    )
    df['industry'] = df.apply(
        lambda x: EnhancedFeatureExtractor.extract_industry_focus(
            x['vacancy_description'],
            lang=x['description_lang']
        ),
        axis=1
    )
    
    lang_reqs = pd.json_normalize(df['language_reqs'])
    df = pd.concat([df, lang_reqs.add_prefix('lang_')], axis=1)
    
    df['relative_seniority'] = df['role'].apply(
        EnhancedFeatureExtractor.calculate_seniority
    )
    
    df['processed_text'] = df.apply(
        lambda x: EnhancedFeatureExtractor.preprocess_text(
            x['vacancy_description'],
            x['description_lang']
        ),
        axis=1
    )
    
    df.drop(columns=['language_reqs'], inplace=True, errors='ignore')
    
    return df

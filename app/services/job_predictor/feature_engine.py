import re
import spacy
import pandas as pd
from langdetect import detect
from typing import Literal, Pattern, Self
from scipy.sparse import csr_matrix, vstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en.stop_words import STOP_WORDS as EN_STOP_WORDS
from spacy.lang.de.stop_words import STOP_WORDS as DE_STOP_WORDS
from .ml_config import (
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
        self.feature_names_ = None
        self.tech_keywords_ = list(TECH_KEYWORDS.keys())
    
    def fit(self: Self, X: pd.DataFrame, y: pd.Series = None) -> Self:
        """Learn the feature names from the tech stack dictionaries."""
        self.feature_names_ = sorted(self.tech_keywords_)
        return self
    
    def transform(self: Self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform tech stack dicts into a binary feature matrix."""
        if self.feature_names_ is None:
            raise ValueError("Must call fit() before transform()")
        
        # Handle both Series and DataFrame input
        if isinstance(X, pd.DataFrame):
            tech_series = X['tech_stack']
        else:
            tech_series = X
        
        # Create a DataFrame with 0/1 values for each tech
        transformed_data = []
        for tech_dict in tech_series:
            row = {tech: 0 for tech in self.feature_names_}
            if isinstance(tech_dict, dict):
                for tech, value in tech_dict.items():
                    if tech in row and value:
                        row[tech] = 1
            transformed_data.append(row)
        
        return pd.DataFrame(transformed_data, columns=self.feature_names_)

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
            stop_words=list(EN_STOP_WORDS)
        )
        self.vectorizer_de = TfidfVectorizer(
            max_features=max_features,
            stop_words=list(DE_STOP_WORDS)
        )
        self.feature_names_ = None
        self.has_en_data = False
        self.has_de_data = False

    def fit(self: Self, X: pd.DataFrame, y=None) -> Self:
        """Fit the vectorizers to English and German text subsets."""
        if not {'processed_text', 'description_lang'}.issubset(X.columns):
            raise ValueError(
                f"Input DataFrame must contain 'processed_text' "
                f"and 'description_lang' columns"
            )
        
        texts_en = X[X['description_lang'] == 'en']['processed_text']
        texts_de = X[X['description_lang'] == 'de']['processed_text']

        self.has_en_data = not texts_en.empty
        self.has_de_data = not texts_de.empty

        if self.has_en_data:
            self.vectorizer_en.fit(texts_en)
        if self.has_de_data:
            self.vectorizer_de.fit(texts_de)

        # Build feature names
        en_features = [
            f"en_{f}" for f in self.vectorizer_en.get_feature_names_out()
        ] if self.has_en_data else []
        
        de_features = [
            f"de_{f}" for f in self.vectorizer_de.get_feature_names_out()
        ] if self.has_de_data else []
 
        self.feature_names_ = en_features + de_features

        return self

    def transform(self: Self, X: pd.DataFrame) -> csr_matrix:
        """Transform text data into a combined TF-IDF feature matrix."""
        results = []
        for _, row in X.iterrows():
            text = row['processed_text']
            lang = row['description_lang']
            
            if lang == 'en' and self.has_en_data:
                results.append(self.vectorizer_en.transform([text]))
            elif lang == 'de' and self.has_de_data:
                results.append(self.vectorizer_de.transform([text]))
            else:
                if self.has_en_data:
                    empty_vec = csr_matrix(
                        (1, len(self.vectorizer_en.get_feature_names_out()))
                    )
                elif self.has_de_data:
                    empty_vec = csr_matrix(
                        (1, len(self.vectorizer_de.get_feature_names_out()))
                    )
                else:
                    empty_vec = csr_matrix((1, 0))
                results.append(empty_vec)

        if results:
            return vstack(results)
        return csr_matrix((len(X), 0))
    
    def get_feature_names_out(self: Self) -> list[str]:
        """Get output feature names with language prefixes."""
        if self.feature_names_ is None:
            raise ValueError("Transformer not fitted yet")
        return self.feature_names_


def create_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering pipeline matching production code."""
    df = df.copy()
    
    df['description_lang'] = df['vacancy_description'].apply(
        lambda x: (
            detect(x[:500]) 
            if isinstance(x, str) and len(x) > 0 
            else 'en')
    )

    def extract_tech_stack(description: str) -> dict:
        if not isinstance(description, str):
            return {}
        return {
            tech: int(bool(re.search(pattern, description.lower())))
            for tech, pattern in TECH_KEYWORDS.items()
        }
    
    df['tech_stack'] = df['vacancy_description'].apply(extract_tech_stack)
    
    def extract_language_requirements(description: str) -> dict:
        if not isinstance(description, str):
            return {
                'german_required': False,
                'english_required': False,
                'german_level': None
            }
        
        requirements = {
            'german_required': (
                bool(LANGUAGE_PATTERNS['german'].search(description))
            ),
            'english_required': (
                bool(LANGUAGE_PATTERNS['english'].search(description))
            ),
            'german_level': None
        }

        if requirements['german_required']:
            german_levels: dict[str, Pattern] = (
                LANGUAGE_PATTERNS['german_levels']
            )
            for level, pattern in german_levels.items():
                if pattern.search(description):
                    requirements['german_level'] = level
                    break

        return requirements
    
    df['language_reqs'] = (
        df['vacancy_description'].apply(extract_language_requirements)
    )    

    def extract_industry_focus(description: str, lang: str = 'en') -> str:
        if not isinstance(description, str):
            return 'other'
        
        description_lower = description.lower()
        patterns = INDUSTRY_PATTERNS.get(lang, INDUSTRY_PATTERNS['en'])

        for industry, pattern in patterns.items():
            if re.search(pattern, description_lower):
                return industry
        return 'other'

    df['industry'] = df.apply(
        lambda x: extract_industry_focus(
            x['vacancy_description'], x['description_lang']
        ),
        axis=1
    )

    def calculate_seniority(role):
        if not isinstance(role, str):
            return 0
        role_lower = role.lower()
        if SENIORITY_PATTERNS['senior'].search(role_lower):
            return 1
        elif SENIORITY_PATTERNS['junior'].search(role_lower):
            return -1
        return 0
    
    df['relative_seniority'] = df['role'].apply(calculate_seniority)
    
    def preprocess_text(text: str, lang: str) -> str:
        """Language-specific preprocessing with error handling."""
        if not isinstance(text, str) or not text.strip():
            return ""
        
        try:
            if lang == 'de' and nlp_de is not None:
                doc = nlp_de(text)
            elif nlp_en is not None:
                doc = nlp_en(text)
            else:
                return simple_tokenize(text)
            
            return " ".join([
                token.lemma_.lower() for token in doc 
                if not token.is_stop and token.is_alpha
            ])
            
        except Exception as e:
            print(f"spaCy processing failed: {e}")
            return simple_tokenize(text)

    def simple_tokenize(text: str) -> str:
        """Fallback tokenization without spaCy."""
        tokens = re.findall(r'\b[a-zA-ZäöüÄÖÜß]+\b', text.lower())

        en_stopwords = {
            'the', 'and', 'is', 'in', 'to', 'of', 'for', 'with', 'on', 'at'
        }
        de_stopwords = {
            'der', 'die', 'das', 'und', 'in', 'zu', 'den', 'von', 'mit', 'für'
        }
        stopwords = en_stopwords.union(de_stopwords)
        
        return " ".join([token for token in tokens if token not in stopwords])

    df['processed_text'] = df.apply(
        lambda x: preprocess_text(
            x['vacancy_description'], x['description_lang']
        ),
        axis=1
    )

    lang_reqs = pd.json_normalize(df['language_reqs'])
    df = pd.concat([df, lang_reqs.add_prefix('lang_')], axis=1)

    df.drop(columns=['language_reqs'], inplace=True, errors='ignore')
    
    return df

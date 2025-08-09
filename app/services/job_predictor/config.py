import re
from pathlib import Path
from typing import Pattern, TypedDict


class GermanLevelPatterns(TypedDict):
    A1: Pattern[str]
    A2: Pattern[str]
    B1: Pattern[str]
    B2: Pattern[str]
    C1: Pattern[str]
    C2: Pattern[str]


class LanguagePatterns(TypedDict):
    german: Pattern[str]
    english: Pattern[str]
    german_levels: GermanLevelPatterns


class SeniorityPatterns(TypedDict):
    senior: Pattern[str]
    junior: Pattern[str]


MODEL_DIR = Path(__file__).parent / 'models'
DEFAULT_MODEL = MODEL_DIR / 'enhanced_model.pkl'


TECH_KEYWORDS = {
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


INDUSTRY_PATTERNS = {
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


LANGUAGE_PATTERNS: LanguagePatterns = {
    'german': re.compile(
        r'(?:fließend|verhandlungssicher|gut|geschäftssicher)\s+deutsch|'
        r'deutsch\s*(?:kenntnisse|erforderlich|voraussetzung|kenntnisse)',
        re.IGNORECASE
    ),
    'english': re.compile(
        r'english\s+(?:fluent|proficient|required|working\s+knowledge)',
        re.IGNORECASE
    ),
    'german_levels': {
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
}


SENIORITY_PATTERNS: SeniorityPatterns = {
    'senior': re.compile(r'\bsenior\b', re.IGNORECASE),
    'junior': re.compile(r'\bjunior\b', re.IGNORECASE)
}

import json
import random
from datetime import datetime, timedelta
from faker import Faker
from pathlib import Path

VALID_STATUSES = {
    "Applied",
    "Interview/Phone", 
    "Interview/Technical",
    "Interview/Onsite",
    "Offer",
    "Rejected",
    "Ghosted"
}

# German job description templates
GERMAN_DESCRIPTIONS = [
    "Wir suchen einen {level} {role} mit {experience} Jahren Erfahrung. {text}",
    "Zur Verstärkung unseres Teams suchen wir einen {level} {role} ({experience} Jahre Erfahrung). {text}",
    "Ihre Herausforderung: Als {role} ({level}) mit {experience} Jahren Erfahrung. {text}",
    "Ihre Aufgaben: Entwicklung und Implementierung als {role} mit {experience}+ Jahren Erfahrung. {text}",
    "Das erwartet Sie: Arbeit als {level} {role} mit mindestens {experience} Jahren Berufserfahrung. {text}"
]

GERMAN_TEXT_SAMPLES = [
    "Sie arbeiten in einem agilen Team und entwickeln innovative Lösungen.",
    "Verantwortung für die Konzeption und Umsetzung von Softwarelösungen.",
    "Entwicklung von skalierbaren Anwendungen in modernen Tech-Stacks.",
    "Zusammenarbeit mit cross-funktionalen Teams in internationalem Umfeld.",
    "Umsetzung von Cloud-basierten Architekturen und Microservices.",
    "Design und Entwicklung von Datenpipelines und ETL-Prozessen.",
    "Implementierung von CI/CD Pipelines und DevOps Practices.",
    "Arbeit mit modernen Frameworks und Entwicklungswerkzeugen.",
    "Qualitätssicherung durch Code Reviews und automatisierten Tests.",
    "Optimierung von Performance und Skalierbarkeit der Anwendungen."
]

def generate_german_description(role, level, experience):
    """Generate a German job description."""
    template = random.choice(GERMAN_DESCRIPTIONS)
    german_text = random.choice(GERMAN_TEXT_SAMPLES)
    return template.format(
        level=level,
        role=role,
        experience=experience,
        text=german_text
    )

def generate_job_application_data(num_records: int = 1000) -> list[dict]:
    """Generate realistic job application test data that matches the data model."""
    fake = Faker()
    
    cities_companies = {
        "Berlin": ["N26", "SoundCloud", "Delivery Hero", "Zalando", "Microsoft Germany"],
        "Munich": ["Siemens", "BMW", "Google Germany", "IBM", "SAP HQ"],
        "Hamburg": ["Xing", "AboutYou", "Airbnb", "Facebook", "Hapag-Lloyd IT"],
        "Frankfurt": ["Deutsche Bank", "Commerzbank", "T-Systems", "Capgemini"],
        "Stuttgart": ["Bosch", "Porsche Digital", "Mercedes-Benz Tech", "IBM Germany"],
        "Cologne": ["DeepL", "Ford Europe IT", "Telekom Deutschland", "Ubisoft Blue Byte"],
        "Düsseldorf": ["Capgemini", "Vodafone Germany", "Accenture Düsseldorf"],
        "Leipzig": ["Amazon", "Dell", "IBM", "Flixbus Tech Hub"],
        "Dresden": ["Infineon", "GlobalFoundries", "Volkswagen IT Solutions"],
        "Karlsruhe": ["SAP", "Fiducia & GAD IT", "BearingPoint IT"],
        "Nuremberg": ["Siemens Healthineers", "Datev", "Adidas Digital Hub"],
        "Bremen": ["OHB Digital", "Airbus Defence IT", "CGI Germany"],
        "Dortmund": ["Adesso", "Signal Iduna IT", "Continental IT"],
        "Potsdam": ["Oracle Germany", "SAP Innovation Center", "Miele Digital"],
        "Saarbrücken": ["SAP AI Research", "DFKI", "Bosch Smart Home"]
    }
    
    sources = [
        "LinkedIn", "XING", "StepStone", "Indeed", "Glassdoor",
        "Stack Overflow Jobs", "Monster", "German Tech Jobs",
        "Make it in Germany", "Company Website", "Recruiter",
        "Europass", "TheLocal", "Honeypot Jobs", "Freelance",
        "Upwork", "EURES", "Berufsstart", "Toptal", "Jobvector"
    ]
    
    roles = ["Data Engineer", "Software Developer"]  
    levels = ["senior", "mid", "junior"]
    experiences = ["3+", "5+", "7+"]
    german_levels = ["A1", "A2", "B1", "B2", "C1", "C2", None]
    
    applications = []
    for i in range(num_records):
        city = random.choice(list(cities_companies.keys()))
        company = random.choice(cities_companies[city])

        applied_date = fake.date_time_between(start_date="-1y", end_date="now")
        response_date = None
        status = random.choice(list(VALID_STATUSES))

        if status not in ["Applied", "Ghosted"]:
            response_date = applied_date + timedelta(days=random.randint(1, 60))
        
        # Calculate response_days if response_date exists
        response_days = None
        if response_date:
            response_days = (response_date - applied_date).days
        
        # Randomly decide if this should be a German job (about 30% chance)
        is_german = random.random() < 0.3
        role = random.choice(roles)
        level = random.choice(levels)
        experience = random.choice(experiences)
        
        if is_german:
            vacancy_description = generate_german_description(role, level, experience)
            description_lang = "de"
        else:
            vacancy_description = (
                f"Seeking {level} {role} with {experience} years experience. "
                f"{fake.text(max_nb_chars=500)}"
            )
            description_lang = "en"
        
        # Create ml_meta structure that matches the data model
        ml_meta = {
            "success_probability": round(random.uniform(0, 1), 2),
            "german_level": random.choice(german_levels),
            "last_prediction_date": datetime.now(),
            "_calculated": {
                "confidence_factor": round(random.uniform(0.5, 1.0), 2),
                "status_weight": round(random.uniform(0.1, 1.0), 2),
                "response_bonus": round(random.uniform(0, 0.2), 2)
            }
        }
        
        application = {
            "company": company,
            "location": city,
            "role": role,
            "status": status,
            "source": random.choice(sources),
            "applied_date": applied_date,
            "response_date": response_date,
            "response_days": response_days,
            "notes": random.choice([
                None,
                f"Recruiter: {fake.name()}",
                f"Tech: {random.choice(['Python', 'Spark', 'SQL'])}",
                f"Ref: {fake.name()}"
            ]),
            "vacancy_description": vacancy_description,
            "description_lang": description_lang,  # Add language field
            "ml_meta": ml_meta,
            "requirements": {
                "german_level": ml_meta["german_level"],
                "tech_stack": random.sample(
                    ["Python", "PostgreSQL", "SQL", "Spark", "Kafka", "MongoDB"],
                    k=random.randint(2, 4)
                )
            }
        }

        applications.append(application)
    
    return applications

def save_to_json(
        data: list[dict],
        filename: str = None
) -> None:
    """Save generated data to JSON file with proper datetime handling."""
    if filename is None:
        project_root = Path(__file__).parent.parent.parent
        filename = project_root / "data" / "samples" / "job_applications.json"

    def json_serializer(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")

    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w") as f:
        json.dump(data, f, indent=2, default=json_serializer)


if __name__ == "__main__":
    sample_data = generate_job_application_data(1000)
    save_to_json(sample_data)
    print(f"Generated {len(sample_data)} application records")
    print(f"German descriptions: {sum(
        1 for app in sample_data if app.get('description_lang') == 'de'
    )}")
    print(f"English descriptions: {sum(
        1 for app in sample_data if app.get('description_lang') == 'en'
    )}")

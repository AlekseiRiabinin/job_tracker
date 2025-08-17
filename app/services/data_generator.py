import json
import random
from datetime import datetime, timedelta
from faker import Faker
from typing import Literal, get_args


type StatusType = Literal[
    "Applied",
    "Interview/Phone",
    "Interview/Technical",
    "Interview/Onsite",
    "Offer",
    "Rejected",
    "Ghosted"
]


def generate_job_application_data(num_records: int = 1000) -> list[dict]:
    """Generate realistic job application test data."""
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
    
    statuses = list(get_args(StatusType))
    
    german_levels = ["A1", "A2", "B1", "B2", "C1", "C2", None]
    
    applications = []
    
    for _ in range(num_records):
        city = random.choice(list(cities_companies.keys()))
        company = random.choice(cities_companies[city])

        applied_date = fake.date_time_between(start_date="-1y", end_date="now")
        response_date = None
        status = random.choice(statuses)
        
        if status not in ["Applied", "Ghosted"]:
            response_date = applied_date + timedelta(days=random.randint(1, 60))
        
        ml_meta = {
            "success_probability": round(random.uniform(0, 1), 2),
            "german_level": random.choice(german_levels),
            "last_prediction_date": datetime.now().isoformat()
        }
        
        application = {
            "company": company,
            "location": city,
            "role": random.choice(roles),
            "status": status,
            "source": random.choice(sources),
            "applied_date": applied_date.isoformat(),
            "response_date": response_date.isoformat() if response_date else None,
            "notes": random.choice([
                None,
                f"Recruiter: {fake.name()}",
                f"Tech: {random.choice(['Python', 'Spark', 'SQL'])}",
                f"Ref: {fake.name()}"
            ]),
            "vacancy_description": (
                f"Seeking {random.choice(['senior', 'mid', 'junior'])} "
                f"{random.choice(roles)} with "
                f"{random.choice(['3+', '5+', '7+'])} years experience. "
                f"{fake.text(max_nb_chars=500)}"
            ),
            "ml": ml_meta,
            "requirements": {
                "german_level": ml_meta["german_level"],
                "tech_stack": random.sample(
                    ["Python", "PodtgreSQL", "SQL", "Spark", "Kafka", "MongoDB"],
                    k=random.randint(2, 4)
                )
            }
        }

        applications.append(application)
    
    return applications

def save_to_json(
        data: list[dict],
        filename: str = "job_applications.json"
) -> None:
    """Save generated data to JSON file."""
    def json_serializer(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")
    
    with open(filename, "w") as f:
        json.dump(data, f, indent=2, default=json_serializer)

if __name__ == "__main__":
    sample_data = generate_job_application_data(1000)
    save_to_json(sample_data)
    print(
        f"Generated {len(sample_data)} "
        f"application records in job_applications.json"
    )

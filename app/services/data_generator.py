import json
import random
from datetime import timedelta
from faker import Faker


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
    
    statuses = [
        "Applied", "Interview/Phone", "Interview/Technical",
        "Interview/Onsite", "Offer", "Rejected", "Ghosted"
    ]
    
    applications = []
    
    for _ in range(num_records):
        city = random.choice(list(cities_companies.keys()))
        company = random.choice(cities_companies[city])

        applied_date = fake.date_time_between(start_date="-1y", end_date="now")
        response_date = None
        status = random.choice(statuses)
        
        if status not in ["Applied", "Ghosted"]:
            response_date = applied_date + timedelta(days=random.randint(1, 60))
        
        notes_options = [
            None,
            f"Recruiter: {fake.name()}",
            f"Tech stack: {random.choice(['Python', 'Spark', 'SQL', 'Hadoop', 'S3'])}",
            f"Reference: {fake.name()}",
            "Requires relocation",
            "Remote position available"
        ]
        
        application = {
            "company": company,
            "location": city,
            "role": random.choice(roles),
            "status": status,
            "source": random.choice(sources),
            "applied_date": applied_date.isoformat(),
            "response_date": response_date.isoformat() if response_date else None,
            "notes": random.choice(notes_options),
            "vacancy_description": (
                f"Looking for {random.choice(['senior', 'mid-level', 'junior'])} "
                f"{random.choice(roles)} with experience in "
                f"{random.choice(['cloud', 'microservices', 'AI', 'big data'])}. "
                f"{fake.text(max_nb_chars=500)}"
            )
        }
        
        applications.append(application)
    
    return applications


def save_to_json(data: list[dict], filename: str = "job_applications.json") -> None:
    """Save generated data to JSON file."""
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    sample_data = generate_job_application_data(1000)
    save_to_json(sample_data)
    print("Generated 1000 job application records in job_applications.json")

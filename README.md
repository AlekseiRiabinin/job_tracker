# Job Tracker Application

A comprehensive Flask-based web application for tracking job applications with MongoDB integration, machine learning capabilities, and Docker containerizat

## Features

- **Job Application Management**: Add, view, update, and delete job applications
- **RESTful API:**: JSON-based API for programmatic access
- **Data Import/Export**: Support for CSV and JSON formats
- **Machine Learning Integration**: Predictive analytics for job application success
- **Multi-language Support**: German and English NLP processing
- **Dockerized**: Easy deployment with Docker and Docker Compose
- **MongoDB Backend**: Scalable NoSQL database storage
- **Analytics Dashboard**: Visual insights into application trends

## Project Structure

```
job_tracker/
├── app/                         # Main application code
│   ├── services/                # Business logic services
│   │   ├── analytics.py         # Analytics and reporting
│   │   ├── data_loader.py       # Data import functionality
│   │   ├── data_exporter.py     # Data export functionality
│   │   ├── data_generator.py    # Test data generation
│   │   └── job_predictor/       # ML prediction services
│   ├── templates/               # HTML templates
│   │   └── dashboard/           # Dashboard views
│   ├── models.py                # Database models
│   ├── routes.py                # Application routes
│   └── cli.py                   # CLI commands
├── migrations/                  # Database migrations
│   └── ml_fields_migration.py   # ML field updates
├── notebooks/                   # Jupyter notebooks
│   ├── ml_training/             # Machine learning experiments
│   ├── analytics/               # Data analysis
│   └── data_validation/         # Data quality checks
├── data/                        # Data directories
│   ├── exports/                 # Exported files (host accessible)
│   └── samples/                 # Import files (place files here)
├── exports/                     # Container exports directory
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container build instructions
└── docker-compose.yml           # Multi-container configuration

```

## Tech stack

### Backend

- **Python 3.12** - Primary programming language
- **Flask** - Web framework with RESTful API support
- **MongoDB 8.0** - NoSQL database for flexible data storage
- **Gunicorn** - Production WSGI HTTP server
- **Pymongo** - MongoDB Python driver

### Machine Learning & NLP

- **spaCy** - Natural Language Processing library
  - `de_core_news_sm` - German language model
  - `en_core_web_sm` - English language model   
- **scikit-learn** - Machine learning algorithms and model training
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computing

### Containerization & Deployment

- **Docker** - Application containerization
- **Docker Compose** - Multi-container orchestration
- **Alpine Linux** - Lightweight base image (slim Python variant)

### Data Processing

- **CSV/JSON** - Data import/export formats
- **Click** - Command-line interface framework
- **Jupyter Notebooks** - Data analysis and ML experimentation

### Development & Monitoring

- **Python-dotenv** - Environment variable management
- **MongoDB Compass** - Database GUI (optional)
- **cURL** - API testing and debugging

### Infrastructure

- **Volume Mounts** - Persistent data storage
- **Docker Networks** - Container communication
- **Health Checks** - Service availability monitoring
- **Resource Limits** - CPU and memory management

## Quick Start

### Prerequisites
- Docker and Docker Compose installed
- Git for version control

### Installation & Deployment

1. **Clone and build the application**
```bash
docker build -t alexflames77/job_tracker:latest .
docker push alexflames77/job_tracker:latest
```

2. **Start the application**
```bash
docker compose up -d
```

3. **Force recreate containers (for updates)**
```bash
docker compose up -d --force-recreate
```

4. **Stop the application**
```bash
docker compose down
```

## API Usage Examples

1. **Get all jobs (JSON format)**

```bash
curl -H "Accept: application/json" http://localhost:5000/
# or
curl http://localhost:5000/?format=json
```

2. **Create a new job application**
```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"company":"Google","role":"Engineer"}' \
     http://localhost:5000/add
```

3. **Delete a job application**
```bash
curl -X DELETE http://localhost:5000/delete/123456789
```

## Data Management

1. **Importing Data**

- From CSV:

```bash
docker exec -it job-tracker flask load-from-csv data/jobs.csv --delimiter=, --encoding=utf-8
```

- From JSON:

```bash
docker exec -it job-tracker flask load-from-json data/jobs.json
```

2. **Exporting Data**

- Export to JSON:

```bash
docker exec -it job-tracker flask export-to-json exports/all_data.json
```

- Export to CSV with query:

```bash
docker exec -it job-tracker flask export-to-csv exports/2024_apps.csv --query '{"applied_date": {"$gte": "2024-01-01"}}'
```

3. **Collection Statistics**

```bash
docker exec -it job-tracker flask collection-stats
```

## Machine Learning Features

### Prediction Flow

- **Form Submission:** Extract job description, role, source, and user German level

- **Feature Engineering:** Create language requirements and skill level features

- **Language Assessment:**

  - Check if German is required

  - Validate user German level against requirements

- **ML Prediction:** Use trained model to predict application success probability

### Running ML Migrations

```bash
docker-compose exec app python migrations/ml_fields_migration.py
```

### Model Management

The application uses spaCy models for NLP processing. Pre-downloaded models include:

- `de_core_news_sm:` German language model
- `en_core_web_sm:` English language model


## Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
FLASK_ENV=production
MONGO_INITDB_ROOT_USERNAME=your_username
MONGO_INITDB_ROOT_PASSWORD=your_password
```

### Volume Mounts

- `./data/samples/` → `/app/data/` (for import files)
- `./data/exports/` → `/app/exports/` (for exported files)


## Development

### Accessing Containers

1. **Shell access to Flask application:**

```bash
docker exec -it job-tracker bash
```

2. **MongoDB access:**

```bash
docker exec -it job_tracker-mongo-1 mongosh -u your_username -p your_password
```

### Testing Data Import

Place your import files in `./data/samples/` on the host machine, then run:

```bash
docker exec -it job-tracker flask load-from-csv data/jobs.csv
```

## Viewing Exported Data

Exported files will be available in ./data/exports/ on the host machine.

## Monitoring & Logs

1. **View application logs:**

```bash
docker compose logs web
```

2. **View database logs:**

```bash
docker compose logs mongo
```

3. **Follow logs in real-time:**

```bash
docker compose logs -f web
```

## Resource Management

The application is configured with resource limits:

- **Web container:** 1 CPU core, 512MB RAM
- **MongoDB container:** 2 CPU cores, 2GB RAM

## Troubleshooting

### Common Issues

1. **Follow logs in real-time:** Ensure port 5000 is available or modify `docker-compose.yml`

2. **MongoDB connection issues:** Check environment variables in `.env` file

3. **Import/export file permissions:** Ensure proper directory permissions for `./data/`

### Health Checks

MongoDB includes health checks that ensure the database is ready before the application starts.

## Contributing

1. Place data files in `./data/samples/`
2. Use the provided notebooks in `./notebooks/` for analysis
3. Follow the existing code structure in `./app/`
4. Test changes with the provided CLI commands

## License

This project is containerized and ready for deployment. Ensure proper licensing for any third-party models or data used.

![Lint Status](https://github.com/YOUR_USERNAME/YOUR_REPO/actions/workflows/lint.yml/badge.svg)
![ML Client CI](https://github.com/YOUR_USERNAME/YOUR_REPO/actions/workflows/ml-client-ci.yml/badge.svg)
![Web App CI](https://github.com/YOUR_USERNAME/YOUR_REPO/actions/workflows/web-app-ci.yml/badge.svg)

# Instrument Classification System

A containerized machine learning application that uses audio classification to identify musical instruments in audio recordings. The system consists of three interconnected Docker containers: a machine learning client that processes audio files using YAMNet, a Flask web application that provides a dashboard for visualizing results, and a MongoDB database that stores all classification results.

## Features

- **Audio Classification**: Uses YAMNet (TensorFlow Hub) and MediaPipe to classify musical instruments in audio files
- **Real-time Dashboard**: Web interface to view classification results, statistics, and recent predictions
- **RESTful API**: Upload audio files via API endpoints for classification
- **Containerized Architecture**: All components run in isolated Docker containers
- **MongoDB Integration**: Persistent storage of all classification results with timestamps and confidence scores

## Team Members


- Siqi Zhu - [https://github.com/HelenZhutt]
- Krystal Lin - [https://github.com/krystalll-0]

## Architecture

The system consists of three main components:

1. **Machine Learning Client** (`machine-learning-client/`): Processes audio files using YAMNet model to detect musical instruments and saves results to MongoDB
2. **Web Application** (`web-app/`): Flask-based web server with dashboard and API endpoints for audio upload and classification
3. **MongoDB Database**: Stores all classification predictions with metadata

## Prerequisites

- Docker and Docker Compose installed
- Python 3.10+ (for local development)
- Git

## Quick Start

### Using Docker Compose (Recommended)

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd 4-containers-qoq
   ```

2. Start all services:
   ```bash
   docker-compose up --build
   ```

3. Access the web application:
   - Open your browser to `http://localhost:5000`
   - The dashboard will be available at `http://localhost:5000/dashboard`

### Manual Setup (Development)

#### 1. Start MongoDB

```bash
docker run --name mongodb -d -p 27017:27017 mongo
```

#### 2. Set Up Machine Learning Client

```bash
cd machine-learning-client
pip install -r requirements.txt

# Set environment variables (optional)
export MONGO_URI="mongodb://localhost:27017"
export MONGO_DB_NAME="ml_logs"
export MONGO_COLLECTION="predictions"
export AUDIO_FILE="data.wav"

# Run the client
python main.py
```

#### 3. Set Up Web Application

```bash
cd web-app
pip install -r requirements.txt

# Set environment variables (optional)
export MONGO_URI="mongodb://localhost:27017"
export MONGO_DB_NAME="ml_logs"
export MONGO_COLLECTION="predictions"
export PORT=5000

# Run the Flask app
python app.py
```

Then open `http://localhost:5000` in your browser.

## Environment Variables

### Machine Learning Client

| Variable | Default | Description |
|----------|---------|-------------|
| `MONGO_URI` | `mongodb://mongodb:27017` | MongoDB connection string |
| `MONGO_DB_NAME` | `ml_logs` | MongoDB database name |
| `MONGO_COLLECTION` | `predictions` | MongoDB collection name |
| `AUDIO_FILE` | `data.wav` | Path to audio file to process |
| `SOURCE_NAME` | `ml-client` | Source identifier for predictions |

### Web Application

| Variable | Default | Description |
|----------|---------|-------------|
| `MONGO_URI` | `mongodb://mongodb:27017` | MongoDB connection string |
| `MONGO_DB_NAME` | `ml_logs` | MongoDB database name |
| `MONGO_COLLECTION` | `predictions` | MongoDB collection name |
| `PORT` | `5000` | Port for Flask web server |
| `AUDIO_MODEL_PATH` | `models/lite-model_yamnet_classification_tflite_1.tflite` | Path to MediaPipe audio model |

### Example `.env` File

Create a `.env` file in the project root (not committed to version control):

```env
# MongoDB Configuration
MONGO_URI=mongodb://mongodb:27017
MONGO_DB_NAME=ml_logs
MONGO_COLLECTION=predictions

# ML Client Configuration
AUDIO_FILE=data.wav
SOURCE_NAME=ml-client

# Web App Configuration
PORT=5000
AUDIO_MODEL_PATH=models/lite-model_yamnet_classification_tflite_1.tflite
```

## MongoDB Setup

### Initial Setup

MongoDB is automatically started when using `docker-compose up`. If running manually:

```bash
docker run --name mongodb -d -p 27017:27017 mongo
```

### Database Structure

The system uses a single collection `predictions` with the following document structure:

```json
{
  "_id": ObjectId("..."),
  "instrument": "guitar",
  "confidence": 0.9234,
  "source": "ml-client",
  "captured_at": ISODate("2024-01-15T10:30:00Z"),
  "created_at": ISODate("2024-01-15T10:30:00Z")
}
```

### Viewing Data

Connect to MongoDB to view stored predictions:

```bash
docker exec -it mongodb mongosh

# Switch to database
use ml_logs

# View all predictions
db.predictions.find().pretty()

# Count predictions
db.predictions.countDocuments()

# Find by instrument
db.predictions.find({instrument: "guitar"})
```

### Starter Data

No starter data is required. The database will be populated automatically as:
- The ML client processes audio files
- Users upload audio files through the web interface

## API Endpoints

### Web Application API

- `GET /` - Landing page
- `GET /dashboard` - Dashboard visualization page
- `GET /api/predictions` - List recent predictions (query param: `limit`, default: 50)
- `POST /api/predictions` - Create a prediction manually (JSON body)
- `POST /api/classify-upload` - Upload and classify an audio file (multipart/form-data, field: `audio`)
- `GET /api/dashboard-data` - Get aggregated dashboard statistics
- `GET /health` - Health check endpoint

### Example: Upload Audio File

```bash
curl -X POST http://localhost:5000/api/classify-upload \
  -F "audio=@path/to/audio.wav"
```

### Example: Create Prediction Manually

```bash
curl -X POST http://localhost:5000/api/predictions \
  -H "Content-Type: application/json" \
  -d '{
    "instrument": "piano",
    "confidence": 0.95,
    "source": "manual",
    "captured_at": "2024-01-15T10:30:00Z"
  }'
```

## Project Structure

```
4-containers-qoq/
├── machine-learning-client/     # ML client subsystem
│   ├── main.py                  # Main ML client code
│   ├── requirements.txt         # Python dependencies
│   ├── yamnet_class_map.csv     # YAMNet class mappings
│   └── data.wav                 # Sample audio file
├── web-app/                     # Web application subsystem
│   ├── app.py                   # Flask application
│   ├── requirements.txt         # Python dependencies
│   ├── models/                  # ML model files
│   ├── static/                  # CSS, JavaScript files
│   └── templates/               # HTML templates
├── test_data/                   # Test audio files
├── docker-compose.yml           # Docker Compose configuration
├── .github/workflows/           # CI/CD workflows
│   ├── lint.yml                # Linting workflow
│   ├── ml-client-ci.yml        # ML client CI (TODO)
│   └── web-app-ci.yml           # Web app CI (TODO)
└── README.md                    # This file
```

## Development

### Code Formatting and Linting

Both subsystems use `black` for formatting and `pylint` for linting:

```bash
# Format code
black machine-learning-client/
black web-app/

# Lint code
pylint machine-learning-client/
pylint web-app/
```

### Running Tests

```bash
# ML Client tests
cd machine-learning-client
pytest --cov=. --cov-report=html

# Web App tests
cd web-app
pytest --cov=. --cov-report=html
```

### CI/CD

The project uses GitHub Actions for continuous integration:

- **Linting**: Runs on every push and pull request
- **ML Client CI**: Builds and tests the ML client (TODO: create workflow)
- **Web App CI**: Builds and tests the web app (TODO: create workflow)

## Troubleshooting

### MongoDB Connection Issues

If the ML client or web app cannot connect to MongoDB:

1. Verify MongoDB is running: `docker ps | grep mongo`
2. Check connection string matches your setup
3. Ensure containers are on the same Docker network (when using docker-compose)

### Port Already in Use

If port 5000 is already in use (common on macOS due to AirPlay):

1. Change the `PORT` environment variable
2. Or disable AirPlay Receiver in System Settings

### Model Loading Errors

If MediaPipe models fail to load:

1. Verify `models/lite-model_yamnet_classification_tflite_1.tflite` exists
2. Check file permissions
3. Ensure sufficient disk space

## License

See [LICENSE](./LICENSE) file for details.

## Contributing

1. Create a feature branch from `main`
2. Make your changes
3. Ensure code passes linting and tests
4. Create a pull request
5. Get code review approval
6. Merge to `main`

## References

- [YAMNet Model](https://tfhub.dev/google/yamnet/1)
- [MediaPipe Audio Classification](https://developers.google.com/mediapipe/solutions/audio/audio_classifier)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [MongoDB Docker](https://www.mongodb.com/compatibility/docker)
- [Docker Compose](https://docs.docker.com/compose/)

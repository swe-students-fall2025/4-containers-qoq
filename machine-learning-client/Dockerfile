FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and data files
COPY main.py .
COPY yamnet_class_map.csv .
COPY data.wav .

# Set default environment variables
ENV MONGO_URI=mongodb://mongodb:27017
ENV MONGO_DB_NAME=ml_logs
ENV MONGO_COLLECTION=predictions
ENV AUDIO_FILE=data.wav
ENV SOURCE_NAME=ml-client

# Run the ML client
CMD ["python", "main.py"]


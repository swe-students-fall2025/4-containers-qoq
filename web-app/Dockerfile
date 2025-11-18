FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY templates/ ./templates/
COPY static/ ./static/
COPY models/ ./models/

# Set default environment variables
ENV MONGO_URI=mongodb://mongodb:27017
ENV MONGO_DB_NAME=ml_logs
ENV MONGO_COLLECTION=predictions
ENV PORT=5000

# Expose port
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]


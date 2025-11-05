FROM python:3.10-slim

WORKDIR /app

# Install system dependencies including git
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install base dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install TimesFM 2.5 from GitHub (not available on PyPI )
RUN pip install --no-cache-dir git+https://github.com/google-research/timesfm.git

# Copy application
COPY app.py .

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "app.py"]

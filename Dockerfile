FROM python:3.12-slim

WORKDIR /app

# Install sys dependencies - ADD libcurl4-openssl-dev HERE
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    cmake \
    llvm \
    llvm-dev \
    libpq-dev \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy project files
COPY . .

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set Python path and make sure numpy is properly installed
RUN python -c "import numpy; numpy.show_config()"

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 8000

# Command to run
CMD ["python", "-u", "src/predict.py"]
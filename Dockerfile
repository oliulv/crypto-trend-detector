# Use an official Python image as the base
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (required for some Python packages)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    cmake \
    llvm-14 \
    llvm-14-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN pip install --upgrade pip

# Copy requirements.txt first to leverage Docker's caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ALL project files into the container
COPY . .

# Set environment variables (loaded from .env later)
ENV PYTHONUNBUFFERED=1

# Command to run when the container starts
CMD ["python", "-u", "predict.py"]
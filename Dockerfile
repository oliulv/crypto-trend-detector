# Use an official Python image as base
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy everything into the container (for now)
COPY . .

# Install dependencies (if requirements.txt exists)
RUN pip install --no-cache-dir -r requirements.txt || true

# Placeholder command to keep the container running
CMD ["python", "-c", "print('Docker container is running...')"]

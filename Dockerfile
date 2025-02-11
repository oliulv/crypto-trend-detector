FROM python:3.10

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    cmake \
    llvm \
    llvm-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# First uninstall any existing numpy
RUN pip uninstall -y numpy

# Install numpy with specific version and all dependencies
#RUN pip install numpy==1.24.3 && \
#    pip install numpy[all]==1.24.3

# Install numpy and scipy with compatible versions
RUN pip install numpy==1.23.5
RUN pip install scipy==1.15.1

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Set Python path and make sure numpy is properly installed
RUN python -c "import numpy; numpy.show_config()"

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Command to run
CMD ["python", "-u", "src/predict.py"]
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files for dependency installation
COPY pyproject.toml .
COPY src/ src/

# Install the package
RUN pip install --no-cache-dir -e .

# Copy remaining files (config, etc.)
COPY . .

EXPOSE 8000

CMD ["uvicorn", "aio.app:app", "--host", "0.0.0.0", "--port", "8000"]

# Use slim Python base image with explicit platform support
FROM --platform=linux/amd64 python:3.10-slim

# Set environment variables to ensure consistent behavior
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create working directory inside container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    poppler-utils \
    git \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Download torch model weights from local models
# Ensure the model folders contain:
#   - config.json
#   - tokenizer_config.json
#   - tokenizer.json
#   - pytorch_model.bin
#   - special_tokens_map.json
#   - vocab.txt or vocab.json
#   - added_tokens.json (if needed)
# Structure:
# /app/models/minilm/
# /app/models/mpnet/

# The script must auto-process PDFs in /app/input and output to /app/output
# Sample: python main.py --input_dir /app/input --output_dir /app/output
ENTRYPOINT ["python", "main.py"]

# Dockerfile for CT-TGNN
# IEEE Q1 Journal Reproducibility

FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    vim \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Install CT-TGNN package
RUN pip install -e .

# Create directories
RUN mkdir -p data logs checkpoints results

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Default command
CMD ["python", "training/trainer.py", "--config", "config/ct_tgnn_config.yaml"]

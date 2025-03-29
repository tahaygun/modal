FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install PyTorch with CUDA support first
RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cu121

# Install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port Gradio will run on
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (if needed by sklearn/pandas)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project source code
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Default command запускает Streamlit
CMD ["streamlit", "run", "src/fraud_detection.py", "--server.address=0.0.0.0"]
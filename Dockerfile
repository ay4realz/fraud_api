# Use lightweight Python
FROM python:3.10-slim

# Set workdir
WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y build-essential

# Copy dependencies
COPY requirements.txt .

# Install deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Expose port
EXPOSE 8000

# Run app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


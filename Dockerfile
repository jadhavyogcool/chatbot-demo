# Use Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy files to container
COPY . /app

# Install dependencies
RUN pip install flask transformers torch

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]

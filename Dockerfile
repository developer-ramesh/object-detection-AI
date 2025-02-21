
# Use the official Python image as the base image
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update

# Set the working directory inside the container
WORKDIR /app


RUN pip install --no-cache-dir ultralytics flask flask-socketio opencv-python numpy eventlet

# Copy the rest of the application code
COPY . .

# Expose port 8000
EXPOSE 8000

# Define the command to run the Flask application
CMD ["python", "app.py"]

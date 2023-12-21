# Use the official Python image from Docker Hub
FROM python:3.9

# Set working directory within the container
WORKDIR /app

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install necessary dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Expose the port the app runs on
EXPOSE 8080

# Define environment variable
ENV FLASK_APP=main.py

# Command to run the application
CMD ["flask", "run", "main"]

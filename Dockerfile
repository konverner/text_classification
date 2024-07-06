# Use the official Python image as the base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install dependencies for git and git-lfs
RUN apt-get update && \
    apt-get install -y git && \
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -y git-lfs && \
    git lfs install && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* \

# Clone the models repository
RUN git clone https://huggingface.co/konverner/lstm_sentiment models/lstm_sentiment

# Copy the pyproject.toml and other necessary files
COPY pyproject.toml .
COPY README.md .
COPY src ./src
COPY tests ./tests
COPY models ./models
COPY notebooks ./notebooks

# Install build dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install project dependencies
RUN pip install --no-cache-dir .

# Install optional dependencies if needed
RUN pip install --no-cache-dir ".[all]"

# Set environment variables for Python
ENV PYTHONUNBUFFERED=1

# Expose port (if needed, adjust according to your application needs)
# EXPOSE 8000

# Define the command to run the application
CMD ["python", "-m", "text_classification"]

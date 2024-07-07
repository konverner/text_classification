# Use the official Python image as the base image
FROM python:3.10-slim
MAINTAINER Konstantin Verner <konst.verner@gmail.com>

# Set the working directory in the container
WORKDIR /app

# Copy the pyproject.toml and other necessary files
COPY pyproject.toml .
COPY src ./src
COPY tests ./tests

# Install build dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install project dependencies
RUN pip install --no-cache-dir .

# Install optional dependencies if needed
RUN pip install --no-cache-dir ".[all]"

# Set environment variables for Python
ENV PYTHONUNBUFFERED=1

# Expose port (if needed, adjust according to your application needs)
EXPOSE 8000

# Clone the models repository
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.rpm.sh | bash
RUN yum install git-lfs -y
RUN git lfs install
RUN git clone "https://huggingface.co/konverner/lstm_sentiment" "models/lstm_sentiment"
RUN rm -rf "models/lstm_sentiment/.git"

# Define the command to run the application
CMD ["python", "src/text_classification/api/main.py"]

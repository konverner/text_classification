# Text Classification API

## Overview

This repository contains a project for text classification using a deep learning model. The repository is organized into several modules, including API, configuration, core functionalities, and tests.


## User Guide

### How to Request the API

To use the text classification API, you can send a POST request to the `/classify` endpoint with a JSON payload containing the text you want to classify.

#### Example Request

```bash
curl -X POST "http://localhost:8000/classify" -H "Content-Type: application/json" -d '{"text": "I love spam"}'
```

#### Example Response

```json
{
    "sentiment": "Positif"
}
```

### API Endpoint Details


| **Endpoint** | `/classify`          |
|--------------|----------------------|
| **Method**   | POST                 |
| **Payload**  | `text` (string): The text you want to classify. |
| **Response** | `sentiment` (string): The predicted classification label. |

## Developer Guide

### Setup and Installation

#### Prerequisites

- Docker
- Python 3.8 or higher

#### Steps to Run the API

**Option 1: Run with Docker**

   - **Pull the Docker image**:

     ```bash
     docker pull konverner/text_classification
     ```

   - **Run the Docker container**:

     ```bash
     docker run -p 8000:8000 konverner/text_classification
     ```

**Option 2: Run with Python Script**

1. **Clone the repository**:

       ```bash
       git clone https://github.com/your-username/text_classification.git
       cd text_classification
       ```

2. **Download the model**:

   Run the provided script to download the model:

   ```bash
   ./scripts/download_model.sh
   ```

3. **Install dependencies**:
    
     ```bash
     pip install .
     ```
    
     or for development:
    
     ```bash
     pip install .[all]
     ```
    
4. Configure the application in [conf/config.yaml](src/text_classification/conf/config.yaml)
    
5. **Run the API**:
    
     ```bash
     python src/text_classification/api/main.py
     ```

### Project Structure

- **notebooks**: Contains Jupyter notebooks for initial experimentation and model training.
- **scripts**: Contains utility scripts such as model download script.
- **src/text_classification/api**: Contains API related code.
  - `classify_router.py`: Defines the routes for the API.
  - `main.py`: Entry point for the API.
  - `schemas.py`: Defines the request and response schemas.
- **src/text_classification/conf**: Contains configuration files.
  - `config.yaml`: Configuration settings.
  - `logging.conf`: Logging configuration.
- **src/text_classification/core**: Contains core functionalities.
  - `classifier.py`: Code for the classifier.
  - `preprocessor.py`: Code for preprocessing text data.
- **tests**: Contains test cases for the project.
  - `units`: Unit tests for API and core functionalities.
  - `bruno`: Test scenarios using Bruno tool.

### Running Tests

To run the unit tests, use the following command:

```bash
pytest tests/units
```

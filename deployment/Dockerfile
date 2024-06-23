FROM --platform=linux/amd64 python:3.11.8-slim

WORKDIR /app

# Dependencies
COPY ./serve-requirements.txt .
RUN pip install --no-cache-dir  -r serve-requirements.txt

# Main script
COPY ./app /app

# Model and tokenizer
COPY ../saved_model /app/saved_model

# Set entrypoint
ENTRYPOINT ["python", "-m", "main"]
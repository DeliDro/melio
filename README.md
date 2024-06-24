# Melio MLOps Machine Translation Challenge

## Author

- **Name**: Déli
- **Given Names**: Dro Kieu
- **GitHub**: [DeliDro](https://github.com/DeliDro)

## Introduction

This project was created for the "Melio MLOps Machine Translation Challenge". It uses the NLLB (No Language Left Behind) model from Meta AI for automatic translation. The goal of this project is to demonstrate the translation capabilities of the NLLB model using a robust and scalable architecture, suitable for MLOps requirements.

## Code Architecture

- **deployment/**: Contains files necessary for deploying and serving the model.
  - **app/**: Contains the application source code.
    - **main.py**: Contains the code to serve the model via kserve.
    - **nllb.py**: Contains the class for translation using the NLLB model.
  - **Dockerfile**: Configuration file to create the Docker image.
  - **docker-compose.yml**: Configuration file to orchestrate Docker containers.
  - **serve_requirements.txt**: List of dependencies needed to serve the model.

- **saved_model/**: Contains the saved model and tokenizer files compatible with `AutoTokenizer` and `AutoModelForSeq2Seq` from the Hugging Face Transformers library.
  - **nllb/**: Directory containing the NLLB model files.
  - **tokenizer/**: Directory containing the tokenizer files.

- **image_name.txt**: Link to the Docker image of the asset on HighwindAI.


## Run
```shell
cd deployment

docker buildx build -t <your_image_name> .

docker compose up -d
```

Be sure to replace `melio_build` by `<your_image_name>` in the `docker-compose.yml` file.

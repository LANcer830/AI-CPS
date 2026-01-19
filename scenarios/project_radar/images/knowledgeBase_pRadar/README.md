# knowledgeBase_pRadar

## Description
This Docker image provides the trained machine learning models for the **Project Radar (pRadar)** AI system.
The image acts as a **knowledge base**, containing pre-trained models that are used during the activation
phase to generate predictions based on unseen input data.

The image does not perform training or inference by itself. It only stores the trained model artifacts
and exposes them via a shared volume to other system components.

## Provided Artifacts
The following model files are provided inside the container at the path `/tmp/knowledgeBase/`:

- `currentAiSolution.h5`  
  Trained Artificial Neural Network (ANN) model created using TensorFlow.
- `currentOlsSolution.pkl`  
  Trained Ordinary Least Squares (OLS) regression model created using Statsmodels.

## Usage Context
This image is intended to be used together with:
- an **activationBase** image providing activation data
- a **codeBase** image executing inference logic

All components are orchestrated via `docker-compose` using a shared external volume.

## Data and Model Origin
The models contained in this image were trained on data scraped from public internet sources
as part of the Project Radar AI project. The data was cleaned, normalized, split into training
and testing datasets, and evaluated before model serialization.

## Course Context
This Docker image was created as part of the course:

**“M. Grum: Advanced AI-based Application Systems”**  
Junior Chair for Business Information Science,  
especially AI-based Application Systems,  
University of Potsdam.

## Ownership
This project was collaboratively developed by the project team members of **Project Radar (pRadar)**.
Each contributor actively participated in the design, implementation, and deployment of the system.

## License
This project and all contained artifacts are released under the  
**GNU Affero General Public License v3.0 (AGPL-3.0)**.

By using this image, you agree to comply with the terms of the AGPL-3.0 license.

# codeBase_pRadar

## Description
This Docker image represents the **execution layer** of the Project Radar (pRadar) AI system.
Its purpose is to apply previously trained machine learning models to unseen activation data
and generate predictions.

The image does not perform model training. Instead, it loads trained models provided by a
separate knowledge base image and applies them to activation data provided by an activation
base image.

## Functional Responsibility
The codeBase image performs the following steps during execution:

1. Reads activation data from `/tmp/activationBase/activation_data.csv`
2. Loads trained AI and OLS models from `/tmp/knowledgeBase/`
3. Applies both models to the activation data
4. Outputs prediction results via standard output (stdout)

This separation follows the AI-CPS architectural principle of decoupling data, knowledge,
and execution logic.

## Dockerfile Explanation
The Dockerfile for this image:

- Uses a lightweight Python base image to enable model loading and execution
- Copies inference code into the container
- Installs required Python dependencies (TensorFlow, pandas)
- Executes the inference script automatically when the container starts

The container is designed to be orchestrated via `docker-compose` and shares its data
exclusively through a mounted external volume.

## Usage Context
This image is intended to be used together with:

- **activationBase_pRadar** – provides activation data
- **knowledgeBase_pRadar** – provides trained models

All components share data via the external Docker volume `ai_system`, mounted at `/tmp`.

## Course Context
This Docker image was created as part of the course:

**“M. Grum: Advanced AI-based Application Systems”**  
Junior Chair for Business Information Science,  
especially AI-based Application Systems,  
University of Potsdam.

## Ownership
This project was collaboratively developed by the Project Radar (pRadar) team.
Each team member contributed to the implementation, testing, and deployment of the system.

## License
This project and all contained artifacts are released under the  
**GNU Affero General Public License v3.0 (AGPL-3.0)**.

By using this image, you agree to comply with the terms of the AGPL-3.0 license.

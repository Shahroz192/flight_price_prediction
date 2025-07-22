# GEMINI.MD: AI Collaboration Guide

This document provides essential context for AI models interacting with this project. Adhering to these guidelines will ensure consistency and maintain code quality.

## 1. Project Overview & Purpose

* **Primary Goal:** Based on the project name "flight_price_prediction", the primary goal is to predict flight prices, likely using machine learning techniques.
* **Business Domain:** The project operates in the Travel and Transportation domain, specifically focusing on the airline industry.

## 2. Core Technologies & Stack

* **Languages:**  It is inferred that Python is the primary programming language, given the user's persona and context. However, the presence of specific Python files or a `requirements.txt` file would confirm this and potentially reveal the Python version used. Confidence level: Medium.
* **Frameworks & Runtimes:** The project likely uses standard Python libraries for data science, such as Pandas, NumPy, and Scikit-learn.  If deep learning is involved, PyTorch might be used. However, without specific file analysis (e.g., `requirements.txt`), this is an inference. Confidence level: Low.
* **Databases:**  No database usage is apparent from the current analysis. The project seems focused on prediction rather than data storage/retrieval. Confidence level: High.
* **Key Libraries/Dependencies:** Key libraries likely include Pandas for data manipulation, NumPy for numerical operations, and Scikit-learn or PyTorch for modeling. Further investigation is needed to confirm. Confidence level: Low.
* **Package Manager(s):** Based on the persona, `uv` or `pip` is the likely package manager. Confidence level: Medium.

## 3. Architectural Patterns

* **Overall Architecture:** Given the project's goal, a typical architecture might involve data preprocessing, feature engineering, model training, and prediction components. This could be structured as a modular Python application. Confidence level: Medium.
* **Directory Structure Philosophy:** The project follows a standard data science structure:
  * `data`: Contains raw, interim, processed, and prediction data.
  * `docs`: Project documentation.
  * `models`: Stores trained models and encoders.
  * `notebooks`: Jupyter notebooks for exploration and analysis.
  * `pipelines`: Orchestration scripts for feature engineering, training, and inference.
  * `src`: Source code for data processing, feature engineering, model training, and prediction.
  * `reports`: For generated reports and figures.
  * `requirements.txt`: Project dependencies.

## 4. Coding Conventions & Style Guide

* **Formatting:** Use ruff for formatting.
* **Naming Conventions:** Assuming Python and PEP 8:
  * `variables`, `functions`: snake_case (`my_variable`)
  * `classes`: PascalCase (`MyClass`)
  * `files`: snake_case or descriptive (`data_preprocessing.py`)
* **API Design:** Not applicable, as this appears to be a prediction model, not an API.
* **Error Handling:**  In Python, typical error handling uses `try...except` blocks. 

## 5. Key Files & Entrypoints

* **Main Entrypoint(s):** The primary entrypoints for the project are the pipeline scripts located in the `pipelines/` directory:
  * `pipelines/feature_pipeline.py`: Runs the data processing and feature engineering steps.
  * `pipelines/training_pipeline.py`: Executes the model training and tuning workflow.
  * `pipelines/inference_pipeline.py`: Runs the prediction pipeline on new data.
* **Configuration:**  Could involve configuration files for model parameters, data paths, etc. These might be simple Python files or use a library like `configparser`. Needs file analysis. Confidence level: Low.
* **CI/CD Pipeline:** No CI/CD pipeline is apparent from the current context.

## 6. Development & Testing Workflow

* **Local Development Environment:**  A Conda environment is recommended for managing dependencies, as stated in the persona. The typical workflow would involve creating and activating a Conda environment, installing dependencies from `requirements.txt`, and running the prediction script. Confidence level: Medium.
* **Testing:**  No testing framework is currently identified. If tests exist, they might use `pytest`. Confidence level: Low.
* **CI/CD Process:**  No CI/CD process is currently identified.

## 7. Specific Instructions for AI Collaboration

* **Contribution Guidelines:**  No contribution guidelines file (`CONTRIBUTING.md`) has been identified.
* **Infrastructure (IaC):**  No IaC directory is present.
* **Security:**  "Be mindful of security. Do not hardcode secrets or keys related to data access or model deployment."
* **Dependencies:** "When adding a new dependency, use `uv add <package>` or `pip install <package>` and update the `requirements.txt` file accordingly."
* **Commit Messages:**  The presence of a `.git` directory suggests the project uses Git for version control. Further analysis of the commit history is needed to understand commit message conventions (e.g., Conventional Commits). Confidence level: Low.

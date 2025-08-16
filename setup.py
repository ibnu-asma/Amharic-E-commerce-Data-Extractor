import os
import shutil

def create_project_structure():
    """Create a modular project structure for the EthioMart project."""
    # Define project directories
    directories = [
        "ethiomart_ner/",
        "ethiomart_ner/src/",
        "ethiomart_ner/src/data_ingestion/",
        "ethiomart_ner/src/preprocessing/",
        "ethiomart_ner/src/labeling/",
        "ethiomart_ner/src/model_training/",
        "ethiomart_ner/src/evaluation/",
        "ethiomart_ner/tests/",
        "ethiomart_ner/notebooks/",
        "ethiomart_ner/data/",
        "ethiomart_ner/data/raw/",
        "ethiomart_ner/data/processed/",
        "ethiomart_ner/data/images/",
        "ethiomart_ner/data/documents/",
        "ethiomart_ner/models/",
        "ethiomart_ner/docs/",
        "ethiomart_ner/.github/",
        "ethiomart_ner/.github/workflows/"
    ]

    # Create directories
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

    # Create __init__.py files for src and its subdirectories, and tests
    init_dirs = [
        "ethiomart_ner/src/",
        "ethiomart_ner/src/data_ingestion/",
        "ethiomart_ner/src/preprocessing/",
        "ethiomart_ner/src/labeling/",
        "ethiomart_ner/src/model_training/",
        "ethiomart_ner/src/evaluation/",
        "ethiomart_ner/tests/"
    ]
    for init_dir in init_dirs:
        with open(f"{init_dir}__init__.py", "w", encoding="utf-8") as f:
            f.write("")
        print(f"Created {init_dir}__init__.py")

    # Create .gitignore file
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*.pyc

# Environment
.env
*.env
venv/

# Data
data/raw/*
data/images/*
data/documents/*
!data/.gitkeep

# Models
models/*
!models/.gitkeep

# Notebooks checkpoints
notebooks/.ipynb_checkpoints/

# Virtual environment
venv/
.env/
"""
    with open("ethiomart_ner/.gitignore", "w", encoding="utf-8") as f:
        f.write(gitignore_content.strip())
    print("Created .gitignore")

    # Create README.md
    readme_content = """
# EthioMart Amharic E-commerce Data Extractor

This project builds a Named Entity Recognition (NER) system to extract key entities (e.g., products, prices, locations) from Amharic Telegram e-commerce channels for EthioMart, a centralized e-commerce platform in Ethiopia.

## Project Structure
```
ethiomart_ner/
├── src/                    # Source code
│   ├── __init__.py
│   ├── data_ingestion/     # Telegram scraping scripts
│   │   ├── __init__.py
│   ├── preprocessing/      # Text preprocessing
│   │   ├── __init__.py
│   │   ├── preprocess.py
│   ├── labeling/           # NER labeling
│   │   ├── __init__.py
│   ├── model_training/     # Model fine-tuning
│   │   ├── __init__.py
│   ├── evaluation/         # Model comparison and interpretability
│   │   ├── __init__.py
├── tests/                  # Unit tests
│   ├── __init__.py
│   ├── test_preprocessing.py
├── notebooks/              # Jupyter notebooks
│   ├── eda.ipynb
├── data/                   # Data storage
│   ├── raw/                # Raw Telegram data
│   │   ├── .gitkeep
│   ├── processed/          # Preprocessed and labeled data
│   │   ├── .gitkeep
│   ├── images/             # Product images
│   │   ├── .gitkeep
│   ├── documents/          # Documents
│   │   ├── .gitkeep
├── models/                 # Trained models
│   ├── .gitkeep
├── docs/                   # Documentation and reports
├── .github/                # GitHub configurations
│   ├── workflows/          # GitHub Actions workflows
│   │   ├── ci.yml
├── requirements.txt        # Python dependencies
├── config.yaml            # Configuration file
├── README.md              # Project overview
└── .gitignore             # Git ignore file
```

## Setup Instructions
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/ethiomart_ner.git
   cd ethiomart_ner
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Telegram API**:
   - Create a `config.yaml` file with your Telegram API credentials:
     ```yaml
     telegram:
       api_id: 'YOUR_API_ID'
       api_hash: 'YOUR_API_HASH'
       phone: 'YOUR_PHONE_NUMBER'
     ```

5. **Run tests**:
   ```bash
   pytest tests/
   ```

6. **Run the project**:
   - Start with the data ingestion script in `src/data_ingestion/`.

## Dependencies
- Python 3.8+
- Telethon
- Pandas
- Hugging Face Transformers
- SHAP
- LIME
- Pytest (for testing)

## License
MIT License
"""
    with open("ethiomart_ner/README.md", "w", encoding="utf-8") as f:
        f.write(readme_content.strip())
    print("Created README.md")

    # Create requirements.txt
    requirements_content = """
telethon==1.36.0
pandas==2.2.3
numpy==1.26.4
transformers==4.44.2
datasets==3.0.1
shap==0.46.0
lime==0.2.0.1
pyyaml==6.0.2
torch==2.4.1
pytest==8.3.3
"""
    with open("ethiomart_ner/requirements.txt", "w", encoding="utf-8") as f:
        f.write(requirements_content.strip())
    print("Created requirements.txt")

    # Create config.yaml
    config_content = """
telegram:
  api_id: 'YOUR_API_ID'
  api_hash: 'YOUR_API_HASH'
  phone: 'YOUR_PHONE_NUMBER'
"""
    with open("ethiomart_ner/config.yaml", "w", encoding="utf-8") as f:
        f.write(config_content.strip())
    print("Created config.yaml")

    # Create GitHub Actions workflow (ci.yml)
    ci_content = """
name: CI Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.8'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Run tests
      run: |
        pytest tests/ --verbose
"""
    with open("ethiomart_ner/.github/workflows/ci.yml", "w", encoding="utf-8") as f:
        f.write(ci_content.strip())
    print("Created .github/workflows/ci.yml")

    # Create sample test file
    test_preprocessing_content = """
import pytest
from src.preprocessing.preprocess import preprocess_amharic_text

def test_preprocess_amharic_text():
    input_text = "ዋጋ 1000 ብር  extra spaces   አዲስ አበባ"
    expected = "ዋጋ 1000 ETB አዲስ አበባ"
    result = preprocess_amharic_text(input_text)
    assert result == expected, f"Expected '{expected}', but got '{result}'"

def test_empty_text():
    assert preprocess_amharic_text("") == "", "Empty text should return empty string"
"""
    with open("ethiomart_ner/tests/test_preprocessing.py", "w", encoding="utf-8") as f:
        f.write(test_preprocessing_content.strip())
    print("Created tests/test_preprocessing.py")

    # Create a sample preprocessing module
    preprocess_content = """
import re
import unicodedata

def preprocess_amharic_text(text):
    if not text:
        return ""
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'(\d+)\s*(birr|ብር)', r'\1 ETB', text, flags=re.IGNORECASE)
    return text
"""
    with open("ethiomart_ner/src/preprocessing/preprocess.py", "w", encoding="utf-8") as f:
        f.write(preprocess_content.strip())
    print("Created src/preprocessing/preprocess.py")

    # Create .gitkeep files for empty directories
    for folder in ["data/raw", "data/processed", "data/images", "data/documents", "models"]:
        with open(f"ethiomart_ner/{folder}/.gitkeep", "w", encoding="utf-8") as f:
            f.write("")
        print(f"Created .gitkeep in {folder}")

    # Create a sample notebook
    notebook_content = """
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# EthioMart NER Project - Exploratory Data Analysis\\n",
    "## Overview\\n",
    "This notebook explores the scraped Telegram data for the EthioMart project."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
"""
    with open("ethiomart_ner/notebooks/eda.ipynb", "w", encoding="utf-8") as f:
        f.write(notebook_content.strip())
    print("Created notebooks/eda.ipynb")

if __name__ == "__main__":
    create_project_structure()
    print("Project structure created successfully!")
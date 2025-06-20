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
   source venv/bin/activate  # On Windows: venv\Scripts\activate
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
# EthioMart Amharic E-commerce Data Extractor

This project builds a Named Entity Recognition (NER) system to extract key entities (e.g., products, prices, locations) from Amharic Telegram e-commerce channels for EthioMart, a centralized e-commerce platform in Ethiopia.

## Project Structure
```
ethiomart_ner/
├── src/                    # Source code
│   ├── __init__.py
│   ├── data_ingestion/     # Telegram scraping scripts
│   ├── preprocessing/      # Text preprocessing
│   ├── labeling/           # NER labeling
│   ├── model_training/     # Model fine-tuning
│   ├── evaluation/         # Model comparison and interpretability
├── tests/                  # Unit tests
├── notebooks/              # Jupyter notebooks (EDA, training, analysis)
├── data/                   # Data storage
│   ├── raw/                # Raw Telegram data
│   ├── processed/          # Preprocessed and labeled data
│   ├── images/             # Product images
│   ├── documents/          # Documents
├── models/                 # Trained models
├── docs/                   # Documentation and reports
├── requirements.txt        # Python dependencies
├── config.yaml             # Configuration file
├── README.md               # Project overview
└── .gitignore              # Git ignore file
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

5. **Run tests** (optional):
   ```bash
   pytest tests/
   ```

---

## Full Workflow: From Data to Vendor Analytics

### 1. **Data Collection**
- **Ingest Telegram posts** using the scripts in `src/data_ingestion/`.
- Output: Raw data in `data/raw/`.

### 2. **Data Preprocessing & Cleaning**
- Use `src/preprocessing/preprocess.py` or the `notebooks/eda.ipynb` notebook to clean and explore the data.
- Output: Cleaned data in `data/processed/`.

### 3. **Data Labeling for NER**
- Annotate a subset of posts for NER using the `src/labeling/label_conll.py` script or manual annotation.
- Output: Labeled data in CoNLL format in `data/labeled/`.

### 4. **Model Training & Fine-Tuning**
- Use `notebooks/ner_training.ipynb` or `notebooks/ner_training_updated.ipynb` to fine-tune transformer models (e.g., DistilBERT, XLM-RoBERTa) on your labeled data.
- Output: Trained model files in `models/`.

### 5. **Model Evaluation & Comparison**
- Compare model performance using `notebooks/model_comparison.ipynb`.
- Evaluate precision, recall, F1-score for each entity type.

### 6. **Model Interpretability**
- Use `notebooks/model_interpretability.ipynb` to analyze model predictions with SHAP and LIME, and understand which tokens influence entity recognition.

### 7. **Vendor Analytics & Scorecard**
- Run `notebooks/vendor_scorecard.ipynb` to:
  - Extract entities from all vendor posts using your best NER model.
  - Combine with post metadata (views, timestamps) to compute:
    - Posting frequency (posts/week)
    - Average views per post
    - Average price point
    - Top performing post
    - Lending score (custom weighted metric)
  - Output: Vendor scorecard table for business analysis, saved in `reports/`.

---

## Example: Running the Full Pipeline

1. **Ingest data:**
   ```bash
   python src/data_ingestion/scraper.py
   ```
2. **Preprocess data:**
   ```bash
   python src/preprocessing/preprocess.py
   ```
3. **Label data:**
   - Use `src/labeling/label_conll.py` or annotate manually.
4. **Train model:**
   - Run cells in `notebooks/ner_training.ipynb` or `notebooks/ner_training_updated.ipynb`.
5. **Evaluate and interpret:**
   - Use `notebooks/model_comparison.ipynb` and `notebooks/model_interpretability.ipynb`.
6. **Vendor analytics:**
   - Run `notebooks/vendor_scorecard.ipynb` for the final business report.

---

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

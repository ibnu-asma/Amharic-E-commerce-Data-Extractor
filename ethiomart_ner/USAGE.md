# EthioMart NER - Separated Scraping and Preprocessing

## Overview
The scraping and preprocessing steps are now decoupled for better flexibility.

## Usage

### Step 1: Scrape Raw Data
```bash
python src/data_ingestion/scraper.py
```
- Scrapes Telegram channels and saves raw JSON files to `data/raw/`
- Downloads media files to `data/raw/media/`
- No preprocessing is performed at this stage

### Step 2: Process Raw Data
```bash
# Process all raw JSON files
python src/preprocessing/process_data.py --all

# Process a specific file
python src/preprocessing/process_data.py --file data/raw/AwasMart_20250621_002946.json

# Default (process all files)
python src/preprocessing/process_data.py
```
- Loads raw JSON files from `data/raw/`
- Applies `preprocess_amharic_text` function
- Saves processed CSV files to `data/processed/`

## File Structure
```
data/
├── raw/
│   ├── *.json          # Raw scraped data
│   └── media/          # Downloaded media files
└── processed/
    └── *.csv           # Processed data with cleaned text
```

## Benefits
- **Flexibility**: Run scraping and preprocessing independently
- **Reprocessing**: Easily reprocess raw data with different preprocessing logic
- **Debugging**: Separate concerns for easier troubleshooting
- **Scalability**: Process large datasets without re-scraping
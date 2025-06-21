# EthioMart NER - Enhanced Data Ingestion System

## Overview
The scraping and preprocessing steps are now decoupled with additional features for better data management.

## Core Usage

### Step 1: Scrape Raw Data
```bash
python src/data_ingestion/scraper.py
```

### Step 2: Process Raw Data
```bash
python src/preprocessing/process_data.py --all
```

## Additional Features

### Data Validation
```bash
python -c "from src.utils.data_validator import validate_processed_data; validate_processed_data('data/processed/AwasMart_20250621_051236.csv')"
```

### Data Statistics
```bash
python src/utils/data_stats.py
```

### Automated Scheduling
```bash
python src/utils/scheduler.py
```
- Scrapes data every 6 hours
- Processes data 30 minutes after scraping

## New Features Added
- **Data Validation**: Quality checks and coverage reports
- **Statistics Generator**: Comprehensive data analysis
- **Automated Scheduler**: Hands-free data collection
- **Entity Extraction**: Basic price, phone, location extraction

## Benefits
- **Flexibility**: Independent scraping and preprocessing
- **Automation**: Scheduled data collection
- **Validation**: Data quality monitoring
- **Analytics**: Comprehensive statistics
- **Scalability**: Process large datasets efficiently